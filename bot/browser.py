"""
bot/browser.py
Playwright-based browser controller for Olymptrade.

Handles:
  • Login with session persistence (avoids re-login every run)
  • Asset selection
  • Fixed Time Trade (FTT) — click UP/DOWN + set duration
  • Forex/CFD trade        — click BUY/SELL + set stop-loss
  • Screenshot on every action for audit trail
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Page, Browser, BrowserContext, TimeoutError as PWTimeout

from config import settings
from models.predictor import Signal

log = logging.getLogger("browser")

ROOT = Path(__file__).resolve().parents[1]
SESSION_FILE = os.path.join(ROOT, "data", "session.json")
SCREENSHOT_DIR = os.path.join(settings.LOG_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)


def _asset_to_url_slug(asset: Optional[str]) -> str:
    """Normalize an asset name to a simple URL slug used by the UI."""
    if not asset:
        return ""
    s = str(asset).strip()
    s = s.replace("/", "-")
    s = s.replace(" ", "-")
    return re.sub(r"[^0-9A-Za-z\-]", "", s).lower()


def _asset_display_name(asset: Optional[str]) -> str:
    """Return the user-facing asset name used by the Olymptrade asset search."""
    if not asset:
        return ""
    asset_str = str(asset).strip()
    has_otc = asset_str.upper().endswith("OTC")
    cleaned = re.sub(r"[^A-Za-z0-9/]+", "", asset_str).upper()

    if "/" in cleaned:
        display = cleaned.replace("/", "/")
    else:
        if len(cleaned) >= 6:
            display = f"{cleaned[:3]}/{cleaned[3:6]}"
        else:
            display = asset_str

    if has_otc and not display.endswith("OTC"):
        display = f"{display} OTC"

    return display


class OlymtradeBot:
    """
    Async Playwright controller.

    Use as an async context manager:

        async with OlymtradeBot() as bot:
            await bot.place_ftt_trade(Signal.BUY, 60)
    """

    def __init__(self):
        self._pw = None
        self._browser: Optional[Browser] = None
        self._ctx: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._logged_in = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def start(self):
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=settings.HEADLESS,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )

        if os.path.exists(SESSION_FILE):
            self._ctx = await self._browser.new_context(
                storage_state=SESSION_FILE,
                viewport={"width": 1366, "height": 768},
            )
            log.info("Loaded saved browser session")
        else:
            self._ctx = await self._browser.new_context(viewport={"width": 1366, "height": 768})

        self._page = await self._ctx.new_page()
        await self._page.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")

    async def stop(self):
        if self._ctx:
            try:
                await self._ctx.storage_state(path=SESSION_FILE)
                log.info("Browser session saved")
            except Exception as e:
                log.warning(f"Could not save browser session: {e}")
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                log.warning(f"Browser close failed during shutdown: {e}")
        if self._pw:
            try:
                await self._pw.stop()
            except Exception as e:
                log.warning(f"Playwright stop failed during shutdown: {e}")

    async def is_alive(self) -> bool:
        """Health check used by the supervisor: True if the page is responsive."""
        try:
            if not self._page or self._page.is_closed():
                return False
            state = await self._page.evaluate("() => document.readyState")
            return state in ("interactive", "complete")
        except Exception:
            return False

    async def login(self) -> bool:
        """Log in with email/password, or fallback to manual Google Sign-In."""
        page = self._page

        try:
            try:
                await page.goto("https://olymptrade.com", wait_until="load", timeout=30_000)
            except PWTimeout:
                log.warning("Page load timed out; falling back to commit.")
                await page.goto("https://olymptrade.com", wait_until="commit", timeout=30_000)
            await asyncio.sleep(2)
        except Exception as e:
            log.error(f"Navigation failed: {e}")
            return False

        try:
            if await page.locator(".platform-chart, .trade-panel, .chart-wrapper").count() > 0:
                log.info("Already logged in (session reused)")
                self._logged_in = True
                return True
        except Exception:
            pass

        log.info("Logging in to Olymptrade...")

        has_credentials = bool(settings.EMAIL and settings.PASSWORD and "@" in settings.EMAIL)

        if has_credentials:
            opened = False
            for selector in [
                'button[data-test="action-sign-in"]',
                'button:has-text("Sign in")',
                'text=Sign in',
                'button:has-text("Log in")',
            ]:
                try:
                    await page.click(selector, timeout=2000, force=True)
                    opened = True
                    await asyncio.sleep(0.6)
                    break
                except Exception:
                    continue

            email_selectors = ['input[type="email"]', 'input[name="email"]', 'input[placeholder*="email"]', 'input[type="text"]']
            found_email = None
            for sel in email_selectors:
                try:
                    await page.wait_for_selector(sel, state="visible", timeout=4000)
                    found_email = sel
                    break
                except Exception:
                    continue

            if not found_email:
                log.warning("Auto-login: email field not found after opening modal")
            else:
                try:
                    await page.fill(found_email, settings.EMAIL, timeout=4000)
                    pass_selectors = ['input[type="password"]', 'input[name="password"]']
                    found_pass = None
                    for ps in pass_selectors:
                        try:
                            await page.wait_for_selector(ps, state="visible", timeout=2000)
                            found_pass = ps
                            break
                        except Exception:
                            continue
                    if found_pass:
                        await page.fill(found_pass, settings.PASSWORD, timeout=3000)
                    await self._screenshot("before_login")
                    for btn in ['button[type="submit"]', '.login-btn', '.sign-in-btn', 'button:has-text("Sign in")', 'button:has-text("Log in")']:
                        try:
                            await page.click(btn, timeout=3000)
                            break
                        except Exception:
                            continue
                    await asyncio.sleep(3)
                    if await page.locator(".platform-chart, .trade-panel, .chart-wrapper").count() > 0:
                        log.info("Automatic login successful!")
                        self._logged_in = True
                        try:
                            await self._ctx.storage_state(path=SESSION_FILE)
                        except Exception:
                            pass
                        return True
                except Exception as e:
                    log.warning(f"Auto-login failed: {e}")

        log.info("MANUAL SIGN-IN if required — waiting up to 180s")
        for i in range(180):
            try:
                url = page.url.lower()
                logged = (
                    await page.locator(".platform-chart, .trade-panel, .chart-wrapper, .sidebar, .user-avatar, [data-test='deposit-button']").count() > 0
                ) or any(k in url for k in ("platform", "cabinet", "dashboard"))
                if logged:
                    log.info(f"Login detected! URL={page.url}")
                    self._logged_in = True
                    try:
                        await self._ctx.storage_state(path=SESSION_FILE)
                    except Exception:
                        pass
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
            if i % 15 == 0 and i > 0:
                log.info(f"Waiting for manual sign-in... ({180-i}s left) | {page.url}")

        log.error("Login timed out (180s)")
        await self._screenshot("login_timeout")
        return False

    async def select_asset(self, asset: str) -> bool:
        page = self._page
        log.info(f"[Browser] Switching to asset: {asset}")
        try:
            for sel in [
                '[data-test="assets-tabs-add-button"]',
                '[data-test="asset-selector"]',
                '[data-test="asset-search"]',
                '[data-testid="asset-selector"]',
                '[data-testid="asset-search"]',
                '.asset-picker',
                '.asset-search',
                '.asset-selection-toggle',
                '.search-assets-button',
                '.asset-open-button',
                'button:has-text("Select Asset")',
                'button:has-text("Asset")',
                'button:has-text("Market")',
            ]:
                try:
                    await page.click(sel, timeout=1500)
                    await asyncio.sleep(0.4)
                    break
                except Exception:
                    continue
            await asyncio.sleep(0.8)

            display_asset = _asset_display_name(asset)
            cleaned = re.sub(r'[^A-Za-z0-9]', '', asset).upper()
            pair = cleaned[:6] if len(cleaned) >= 6 else cleaned
            slash_pair = f"{pair[:3]}/{pair[3:]}" if len(pair) == 6 else asset
            space_pair = f"{pair[:3]} {pair[3:]}" if len(pair) == 6 else asset

            search_tokens = [
                display_asset,
                asset,
                asset.upper(),
                asset.replace('-', ' '),
                asset.replace('/', ' '),
                asset.replace('-', ''),
                asset.replace('/', ''),
                asset.replace('-', '/'),
                asset.replace('-', '/ '),
                asset.replace('/', '-'),
                slash_pair,
                f"{slash_pair} OTC",
                f"{slash_pair}-OTC",
                f"{space_pair} OTC",
                f"{space_pair}-OTC",
                pair,
            ]
            search_queries = [q for q in dict.fromkeys([q for q in search_tokens if q])]
            search_selectors = [
                'input[placeholder*="Search"]',
                'input[placeholder*="search"]',
                'input[type="search"]',
                'input[aria-label*="Search"]',
                'input[aria-label*="search"]',
                'input[role="searchbox"]',
                'input[placeholder*="Asset"]',
                'input[placeholder*="Symbol"]',
                'input[placeholder*="Pair"]',
                '.asset-search input',
                '.search-input input',
                '.asset-search-field input',
                '.search-field input',
                'input',
            ]
            search_input = None
            for s in search_selectors:
                try:
                    locator = page.locator(s)
                    if await locator.count() > 0:
                        first = locator.first
                        if await first.is_visible():
                            search_input = first
                            break
                except Exception:
                    continue

            if search_input:
                for q in [display_asset] + search_queries:
                    try:
                        await search_input.click(timeout=1200)
                        await search_input.fill('', timeout=1200)
                        await search_input.type(q, delay=10)
                        await asyncio.sleep(0.4)
                        await page.keyboard.press('Enter')
                        await asyncio.sleep(0.8)
                        typed = await search_input.input_value()
                        if typed and (typed == q or q in typed):
                            break
                    except Exception:
                        continue
            else:
                log.warning('[Browser] Asset search input not found')

            await asyncio.sleep(0.8)

            variants = list(dict.fromkeys([
                display_asset,
                asset,
                asset.upper(),
                asset.replace('-', ' '),
                asset.replace('/', ' '),
                asset.replace('-', ''),
                asset.replace('/', ''),
                slash_pair,
                f"{slash_pair} OTC",
                f"{slash_pair}-OTC",
                space_pair,
                f"{space_pair} OTC",
                pair,
            ]))

            for v in variants:
                for selector in [
                    f'button:has-text("{v}")',
                    f'a:has-text("{v}")',
                    f'.assets-list-item:has-text("{v}")',
                    f'.asset-item:has-text("{v}")',
                    f'.asset-row:has-text("{v}")',
                    f'.search-result-item:has-text("{v}")',
                    f'li:has-text("{v}")',
                    f'span:has-text("{v}")',
                ]:
                    try:
                        locator = page.locator(selector)
                        if await locator.count() > 0:
                            await locator.first.click(timeout=1200)
                            await asyncio.sleep(0.6)
                            log.info(f"[Browser] ✓ Asset {asset} selected via text match")
                            return True
                    except Exception:
                        continue

            try:
                try:
                    await page.evaluate("(asset) => {"
                                        "try{ localStorage.setItem('SELECTED_AGGREGATE', JSON.stringify({mechanic:'ftt', assetId: asset})); }catch(e){};"
                                        "try{ localStorage.setItem('SELECTED_ASSET', asset); }catch(e){};"
                                        "window.dispatchEvent(new Event('storage'));"
                                        "}", asset)
                except Exception:
                    pass

                clicked = await page.evaluate("(asset) => {\n"
                                             "const normalize = (s) => s ? String(s).toUpperCase().replace(/[-/ ]+/g,' ').trim() : '';\n"
                                             "const target = normalize(asset);\n"
                                             "const candidates = [asset, asset.replace(/-/g,' '), asset.replace(/\\//g,' '), asset.replace(/[-/ ]+/g,''), target, target.replace(/ /g,'')];\n"
                                             "const nodes = Array.from(document.querySelectorAll('button, a, li, div, span'));\n"
                                             "for(const node of nodes){ if(node.innerText && candidates.some(t => normalize(node.innerText).includes(normalize(t)))){ node.click(); return true; }}\n"
                                             "const byAttr = document.querySelector(`[data-asset-id='${asset}'], [data-asset-id='${asset.replace(/-/g,'')}'], [data-asset-id='${asset.replace(/\\//g,'')}'], [data-asset-id='${asset.replace(/[-/ ]+/g,'')}']`);\n"
                                             "if(byAttr){ byAttr.click(); return true; }\n"
                                             "return false;\n"
                                             "}", asset)
                if clicked:
                    await asyncio.sleep(0.6)
                    log.info(f"[Browser] ✓ Asset {asset} selected via DOM evaluation")
                    return True
            except Exception:
                pass

            try:
                await page.click('.assets-list-item, .asset-item, .asset-row, .search-result-item', timeout=1200)
                await asyncio.sleep(0.6)
                log.info(f"[Browser] ✓ Asset {asset} selected via fallback click")
                return True
            except Exception:
                pass

            log.warning(f"[Browser] ✗ Could not select asset {asset} — all methods failed")
        except Exception as e:
            log.warning(f"Asset selection failed: {e}")
            return False

    async def _verify_chart_loaded(self, asset: str) -> bool:
        page = self._page
        if not page:
            return False
        try:
            display_asset = _asset_display_name(asset)
            result = await page.evaluate("(asset, displayAsset) => {\n"
                                         "const normalize = (s) => s ? String(s).toUpperCase().replace(/[-/ ]+/g,' ').trim() : '';\n"
                                         "const target = normalize(displayAsset || asset);\n"
                                         "const nodes = Array.from(document.querySelectorAll('span, div, button, a, h1, h2, h3, p, strong'));\n"
                                         "return nodes.some(node => node.innerText && normalize(node.innerText).includes(target));\n"
                                         "}", asset, display_asset)
            return bool(result)
        except Exception:
            return False

    async def set_amount(self, amount: float):
        page = self._page
        for sel in ['input[data-test="deal-amount-input"]', 'input[data-test="amount"]', '.amount-input input', 'input.trade-amount']:
            try:
                await page.fill(sel, str(int(amount)), timeout=2000)
                await asyncio.sleep(0.2)
                return
            except Exception:
                continue
        log.warning("Could not set trade amount")

    async def place_ftt_trade(self, signal: Signal, duration_seconds: int, asset: str = None) -> bool:
        page = self._page
        try:
            if asset:
                is_correct = await self._verify_chart_loaded(asset)
                if not is_correct:
                    log.info(f"[Trade] Asset {asset} not currently open. Switching...")
                    switched = await self.select_asset(asset)
                    if not switched:
                        log.error(f"[Trade] Aborting trade: could not switch to asset {asset}")
                        return False

            await self._switch_trade_mode('ftt')
            await self._set_ftt_duration(duration_seconds)
            await self.set_amount(settings.TRADE_AMOUNT)
            
            sig_val = getattr(signal, "value", str(signal)).upper()
            await self._screenshot(f"ftt_before_{sig_val.lower()}")
            
            if sig_val == "BUY":
                buy_btn = page.locator('button[data-test="deal-button-up"]')
                if await buy_btn.count() > 0:
                    await buy_btn.first.click(timeout=5000)
                else:
                    await page.locator('button:has-text("Up")').first.click(timeout=5000)
                log.info(f"[Trade] FTT UP placed | {duration_seconds}s | Asset: {asset}")
            else:
                sell_btn = page.locator('button[data-test="deal-button-down"]')
                if await sell_btn.count() > 0:
                    await sell_btn.first.click(timeout=5000)
                else:
                    await page.locator('button:has-text("Down")').first.click(timeout=5000)
                log.info(f"[Trade] FTT DOWN placed | {duration_seconds}s | Asset: {asset}")
            await asyncio.sleep(1)
            await self._screenshot(f"ftt_after_{sig_val.lower()}")
            return True
        except Exception as e:
            log.error(f"[Trade] FTT trade failed: {e}")
            await self._screenshot('ftt_error')
            return False

    async def _set_ftt_duration(self, seconds: int):
        page = self._page
        labels = self._ftt_duration_labels(seconds)
        for label in labels:
            try:
                await page.click(f'button:has-text("{label}")', timeout=1200)
                await asyncio.sleep(0.2)
                return
            except Exception:
                continue

    def _ftt_duration_labels(self, seconds: int) -> list[str]:
        labels: list[str] = []
        if seconds >= 60:
            mins = seconds // 60
            labels.extend([f"{mins} min", f"{mins}m"])
            labels.extend([f"{seconds} sec", f"{seconds}s"])
        else:
            labels.extend([f"{seconds} sec", f"{seconds}s"])
        if seconds == 60:
            labels = ["1 min", "1m", "60 sec", "60s"] + labels
        return list(dict.fromkeys(labels))

    async def place_forex_trade(self, signal: Signal, asset: str = None) -> bool:
        page = self._page
        try:
            await self._switch_trade_mode('forex')
            await self.set_amount(settings.TRADE_AMOUNT)
            await self._screenshot(f'forex_before_{signal.value.lower()}')
            if signal == Signal.BUY:
                await page.locator('button[data-test="buy-btn"], button:has-text("Buy")').first.click(timeout=5000)
                log.info(f"[Trade] Forex BUY placed | Asset: {asset}")
            else:
                await page.locator('button[data-test="sell-btn"], button:has-text("Sell")').first.click(timeout=5000)
                log.info(f"[Trade] Forex SELL placed | Asset: {asset}")
            await asyncio.sleep(0.8)
            await self._screenshot(f'forex_after_{signal.value.lower()}')
            try:
                if await page.locator('button:has-text("Confirm"), .confirm-btn').count() > 0:
                    await page.locator('button:has-text("Confirm"), .confirm-btn').first.click()
            except Exception:
                pass
            return True
        except Exception as e:
            log.error(f"[Trade] Forex trade failed: {e}")
            await self._screenshot('forex_error')
            return False

    async def _set_forex_stop_loss(self):
        page = self._page
        try:
            sl = page.locator('input[data-test="stop-loss"], .stop-loss-input')
            if await sl.count() > 0:
                await sl.first.fill(str(settings.STOP_LOSS_PCT))
        except Exception:
            pass

    async def _switch_trade_mode(self, mode: str):
        page = self._page
        try:
            if mode == 'ftt':
                sel = 'button:has-text("Fixed Time"), [data-test="ftt-tab"]'
            else:
                sel = 'button:has-text("Forex"), [data-test="forex-tab"]'
            try:
                await page.click(sel, timeout=1500)
                await asyncio.sleep(0.6)
            except Exception:
                pass
        except Exception:
            pass

    async def _screenshot(self, label: str):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"{ts}_{label}.png")
            if self._page:
                await self._page.screenshot(path=path, timeout=5000)
        except Exception:
            pass

    async def draw_zone_lines(
        self,
        upper: float,
        lower: float,
        pair: str = "",
        confirmation: Optional[str] = None,
    ) -> None:
        """
        Injects a persistent canvas overlay onto the Olymp Trade live chart.
        Draws the upper and lower boundary lines for the 4H order zone plus
        a shaded rectangle between them.
        """
        page = self._page
        if not page:
            log.warning("[LiveChart] draw_zone_lines skipped: page not initialized")
            return

        confirmation_text = confirmation or "WAITING"
        confirmation_js = json.dumps(confirmation_text)
        js = f"""
(function() {{
    const old = document.getElementById('ftt_zone_overlay');
    if (old) old.remove();
    if (window._ftt_zone_anim) cancelAnimationFrame(window._ftt_zone_anim);

    const UPPER = {upper};
    const LOWER = {lower};
    const PAIR  = '{pair}';
    const CONFIRMATION = {confirmation_js};

    function getChartCanvas() {{
        const selectors = [
            'canvas.chart__canvas',
            '.chart-area canvas',
            '.trading-chart canvas',
            'canvas[class*="chart"]',
            'canvas'
        ];
        for (const sel of selectors) {{
            const el = document.querySelector(sel);
            if (el && el.width > 200 && el.height > 100) return el;
        }}
        return null;
    }}

    const baseCanvas = getChartCanvas();
    if (!baseCanvas) {{
        console.warn('[FTT] No chart canvas found');
        return;
    }}

    const overlay = document.createElement('canvas');
    overlay.id = 'ftt_zone_overlay';
    overlay.style.position = 'absolute';
    overlay.style.pointerEvents = 'none';
    overlay.style.zIndex = '9999';
    overlay.style.top    = baseCanvas.offsetTop  + 'px';
    overlay.style.left   = baseCanvas.offsetLeft + 'px';
    overlay.width  = baseCanvas.width;
    overlay.height = baseCanvas.height;
    baseCanvas.parentElement.style.position = 'relative';
    baseCanvas.parentElement.appendChild(overlay);

    const ctx = overlay.getContext('2d');

    function getPriceRange() {{
        const labels = document.querySelectorAll(
            '.chart-price-axis text, .y-axis text, [class*="price-label"], [class*="yAxis"] text'
        );
        let prices = [];
        labels.forEach(el => {{
            const v = parseFloat(el.textContent.replace(/[^0-9.]/g, ''));
            if (!isNaN(v) && v > 0) prices.push(v);
        }});
        if (prices.length >= 2) {{
            return {{ min: Math.min(...prices), max: Math.max(...prices) }};
        }}
        const pad = (UPPER - LOWER) * 3;
        return {{ min: LOWER - pad, max: UPPER + pad }};
    }}

    function priceToY(price, pmin, pmax, h) {{
        return h - ((price - pmin) / (pmax - pmin)) * h;
    }}

    function draw() {{
        const W = overlay.width;
        const H = overlay.height;
        ctx.clearRect(0, 0, W, H);

        const {{ min: pmin, max: pmax }} = getPriceRange();
        if (pmax <= pmin) return;

        const yUpper = priceToY(UPPER, pmin, pmax, H);
        const yLower = priceToY(LOWER, pmin, pmax, H);

        const visible = (v) => v >= -10 && v <= H + 10;
        if (!visible(yUpper) && !visible(yLower)) return;

        ctx.fillStyle = 'rgba(100, 180, 255, 0.07)';
        ctx.fillRect(0, Math.min(yUpper, yLower), W,
                     Math.abs(yLower - yUpper));

        if (visible(yUpper)) {{
            ctx.beginPath();
            ctx.setLineDash([8, 4]);
            ctx.strokeStyle = 'rgba(220, 50, 50, 0.92)';
            ctx.lineWidth   = 2;
            ctx.moveTo(0, yUpper);
            ctx.lineTo(W, yUpper);
            ctx.stroke();

            ctx.setLineDash([]);
            ctx.fillStyle    = 'rgba(220, 50, 50, 0.9)';
            ctx.font         = 'bold 11px monospace';
            ctx.textAlign    = 'left';
            ctx.fillText(`▲ HH {upper:.5f}  SUPPLY`, 6, yUpper - 4);
        }}

        if (visible(yLower)) {{
            ctx.beginPath();
            ctx.setLineDash([8, 4]);
            ctx.strokeStyle = 'rgba(40, 200, 80, 0.92)';
            ctx.lineWidth   = 2;
            ctx.moveTo(0, yLower);
            ctx.lineTo(W, yLower);
            ctx.stroke();

            ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(40, 200, 80, 0.9)';
            ctx.font      = 'bold 11px monospace';
            ctx.textAlign = 'left';
            ctx.fillText(`▼ LL {lower:.5f}  DEMAND`, 6, yLower + 13);
        }}

        ctx.fillStyle   = 'rgba(20, 20, 30, 0.75)';
        ctx.fillRect(8, 8, 310, 80);
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth   = 1;
        ctx.setLineDash([]);
        ctx.strokeRect(8, 8, 310, 80);

        ctx.font      = 'bold 12px monospace';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'left';
        ctx.fillText(`FTT ZONE — ${{PAIR}}`, 16, 26);

        ctx.fillStyle = 'rgba(220, 50, 50, 0.95)';
        ctx.fillText(`UPPER (HH): {upper:.5f}`, 16, 42);

        ctx.fillStyle = 'rgba(40, 200, 80, 0.95)';
        ctx.fillText(`LOWER (LL): {lower:.5f}`, 16, 58);

        ctx.fillStyle = 'rgba(255, 215, 0, 0.95)';
        ctx.fillText(`CONFIRM: ${{CONFIRMATION}}`, 16, 74);
    }}

    function loop() {{
        draw();
        window._ftt_zone_anim = requestAnimationFrame(loop);
    }}
    loop();

    console.log('[FTT] Zone lines drawn: UPPER={upper} LOWER={lower}');
}})();
"""
        try:
            await page.evaluate(js)
            log.info(
                f"[LiveChart] Zone lines drawn on website | "
                f"pair={pair} upper={upper:.5f} lower={lower:.5f} confirmation={confirmation_text}"
            )
        except Exception as e:
            log.warning(f"[LiveChart] draw_zone_lines failed: {e}")

    async def clear_zone_lines(self) -> None:
        """Remove the zone line overlay from the chart."""
        page = self._page
        if not page:
            return

        js = """
(function() {
    const el = document.getElementById('ftt_zone_overlay');
    if (el) el.remove();
    if (window._ftt_zone_anim) {
        cancelAnimationFrame(window._ftt_zone_anim);
        window._ftt_zone_anim = null;
    }
    console.log('[FTT] Zone lines cleared');
})();
"""
        try:
            await page.evaluate(js)
            log.info("[LiveChart] Zone lines cleared from website")
        except Exception as e:
            log.debug(f"[LiveChart] clear_zone_lines: {e}")

    async def close_active_position(self, signal: Signal, entry_price: float) -> bool:
        page = self._page
        try:
            rows = ['.position-row', '.open-position', '.deal-row']
            for r in rows:
                try:
                    nodes = await page.locator(r).all()
                    for n in nodes:
                        try:
                            txt = await n.inner_text()
                            if (signal.value in txt or signal.name in txt) and str(entry_price)[:6] in txt:
                                try:
                                    btn = n.locator('button:has-text("Close"), button.close-btn')
                                    if await btn.count() > 0:
                                        await btn.first.click()
                                        await asyncio.sleep(0.5)
                                        await self._screenshot('early_close')
                                        return True
                                except Exception:
                                    continue
                        except Exception:
                            continue
                except Exception:
                    continue

            for sel in ['button:has-text("Close"), .close-position, .close-deal, button.close']:
                try:
                    await page.click(sel, timeout=1200)
                    await asyncio.sleep(0.4)
                    await self._screenshot('early_close_fallback')
                    return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    async def get_current_price(self) -> Optional[float]:
        def is_plausible(v: float) -> bool:
            return 0.01 <= v <= 50.0

        def clean_text(txt: str) -> Optional[float]:
            try:
                text = txt.replace(',', '').replace(' ', '')
                # Accept only values that look like prices, not percentages or counts.
                if not re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]+)?", text):
                    return None
                v = float(text)
                return v if is_plausible(v) else None
            except Exception:
                return None

        try:
            title = await self._page.title()
            import re
            m = re.search(r"([0-9]{1,3}(?:[.,][0-9]{1,6})?)", title)
            if m:
                v = clean_text(m.group(1).replace(',', '.'))
                if v is not None:
                    return v
        except Exception:
            pass

        selectors = [
            '.current-price',
            '.bid-price',
            '.asset-price',
            '.price-value',
            '[data-testid*="price"]',
            '[class*="price"]',
            '[class*="asset"]',
        ]

        for sel in selectors:
            try:
                el = self._page.locator(sel)
                if await el.count() == 0:
                    continue
                for i in range(min(await el.count(), 6)):
                    txt = await el.nth(i).inner_text()
                    v = clean_text(txt)
                    if v is not None:
                        return v
            except Exception:
                continue

        try:
            payload = await self._page.evaluate(r'''() => {
                const text = document.body.innerText || '';
                const matches = text.match(/\b(\d{1,3}(?:\.\d{1,6})?)\b/g) || [];
                for (const m of matches) {
                    const v = parseFloat(m);
                    if (v >= 0.01 && v <= 50) return v;
                }
                return null;
            }''')
            if payload is not None and is_plausible(payload):
                return payload
        except Exception:
            pass

        return None

    async def get_asset_payout(self, asset: str) -> Optional[int]:
        """Read the current payout percentage for the selected asset from the platform UI."""
        if not self._page:
            return None

        try:
            display_asset = _asset_display_name(asset)
            await self._verify_chart_loaded(asset)
            payload = await self._page.evaluate(r'''(displayAsset) => {
                const normalize = (value) => value ? String(value).replace(/[\n\t\r]+/g,' ').trim() : '';
                const nodes = Array.from(document.querySelectorAll('span, div, button, a, p, strong'));
                for (const node of nodes) {
                    const text = normalize(node.innerText);
                    const match = text.match(/(\d{2,3})\s*%/);
                    if (match) {
                        return match[1];
                    }
                }
                return null;
            }''', display_asset)
            if payload is None:
                return None
            return int(payload)
        except Exception:
            return None
