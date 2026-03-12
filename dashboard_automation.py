"""
Dashboard Automation Script
Navigates to the LVC dashboard and captures screenshots while interacting with it.
"""
import asyncio
import time
from playwright.async_api import async_playwright
from pathlib import Path

async def main():
    # Create screenshots directory
    screenshots_dir = Path("dashboard_automation_screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    async with async_playwright() as p:
        # Launch browser
        print("Launching browser...")
        browser = await p.chromium.launch(
            headless=False,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        print("Creating context...")
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True
        )
        print("Creating page...")
        page = await context.new_page()
        print("Browser ready!")
        
        print("STEP 1: Navigating to http://127.0.0.1:8051/")
        await page.goto('http://127.0.0.1:8051/', timeout=60000)
        
        # Wait for page to load
        try:
            await page.wait_for_load_state('domcontentloaded', timeout=30000)
        except:
            print("  Note: Page may still be loading, continuing anyway...")
        await asyncio.sleep(3)
        
        # Take initial screenshot
        screenshot_path = screenshots_dir / "01_initial_page.png"
        await page.screenshot(path=str(screenshot_path), full_page=False)
        print(f"✓ Screenshot saved: {screenshot_path}")
        
        print("\nSTEP 2: Looking for week selector dropdown...")
        
        # Try to find the dropdown - common Dash dropdown selectors
        dropdown_selectors = [
            'div[id*="week"]',
            'div[id*="target"]',
            'div.dash-dropdown',
            '.Select-control',
            'div[class*="Select"]'
        ]
        
        dropdown = None
        for selector in dropdown_selectors:
            try:
                dropdown = await page.query_selector(selector)
                if dropdown:
                    print(f"✓ Found dropdown with selector: {selector}")
                    break
            except:
                continue
        
        if not dropdown:
            print("⚠ Could not find dropdown, trying to get page content...")
            # Get all dropdowns on page
            all_dropdowns = await page.query_selector_all('.Select-control')
            if all_dropdowns:
                dropdown = all_dropdowns[0]
                print(f"✓ Found {len(all_dropdowns)} dropdown(s), using first one")
        
        if dropdown:
            # Click on dropdown to open it
            print("Clicking dropdown to open options...")
            await dropdown.click()
            await asyncio.sleep(2)
            
            # Take screenshot of opened dropdown
            screenshot_path = screenshots_dir / "02_dropdown_opened.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"✓ Screenshot saved: {screenshot_path}")
            
            # Look for the 2026-02-22 option
            print("Looking for 2026-02-22 option...")
            
            # Wait for options to appear
            await page.wait_for_selector('.Select-option, .VirtualizedSelectOption', timeout=5000)
            await asyncio.sleep(1)
            
            # Get all options and print them
            print("Checking available options...")
            options = await page.query_selector_all('.Select-option, .VirtualizedSelectOption, div[role="option"]')
            
            if not options:
                # Try alternative selectors
                options = await page.query_selector_all('.Select-menu-outer div, [class*="option"]')
            
            print(f"Found {len(options)} option(s)")
            option_found = False
            
            for i, opt in enumerate(options):
                try:
                    text = await opt.inner_text()
                    text = text.strip()
                    print(f"  Option {i+1}: '{text}'")
                    if "02-22" in text or "2026-02-22" in text or "2/22" in text:
                        print(f"  ✓ Clicking this option!")
                        await opt.click()
                        option_found = True
                        break
                except Exception as e:
                    print(f"  Error reading option {i+1}: {e}")
                    continue
            
            if option_found:
                print("✓ Selected 2026-02-22 option")
                
                print("\nSTEP 3: Waiting for dashboard to refresh...")
                await asyncio.sleep(5)
                
                # Take screenshot after selection
                screenshot_path = screenshots_dir / "03_after_week_selection.png"
                await page.screenshot(path=str(screenshot_path), full_page=False)
                print(f"✓ Screenshot saved: {screenshot_path}")
            else:
                print("⚠ Could not find or click 2026-02-22 option")
        else:
            print("⚠ Could not find dropdown element")
        
        print("\nSTEP 4: Scrolling through entire page...")
        
        # Get page height
        page_height = await page.evaluate('document.body.scrollHeight')
        viewport_height = 1080
        scroll_positions = list(range(0, page_height, viewport_height // 2))
        
        for i, scroll_pos in enumerate(scroll_positions):
            await page.evaluate(f'window.scrollTo(0, {scroll_pos})')
            await asyncio.sleep(1)
            
            screenshot_path = screenshots_dir / f"04_scroll_{i+1:02d}_pos_{scroll_pos}.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"✓ Screenshot saved: {screenshot_path} (scroll position: {scroll_pos}px)")
        
        # Scroll back to top
        await page.evaluate('window.scrollTo(0, 0)')
        await asyncio.sleep(1)
        
        print("\nSTEP 5: Looking for 'Vintage Deep Dive' tab...")
        
        # Try to find the tab
        tab_selectors = [
            'text="Vintage Deep Dive"',
            'text=/.*Vintage Deep Dive.*/',
            '.tab:has-text("Vintage Deep Dive")',
            'button:has-text("Vintage Deep Dive")',
            'a:has-text("Vintage Deep Dive")',
            '[role="tab"]:has-text("Vintage Deep Dive")'
        ]
        
        tab_found = False
        for selector in tab_selectors:
            try:
                tab = await page.query_selector(selector)
                if tab:
                    print(f"✓ Found tab with selector: {selector}")
                    await tab.click()
                    tab_found = True
                    await asyncio.sleep(2)
                    break
            except Exception as e:
                continue
        
        if not tab_found:
            # Get all tabs and print them
            print("Could not find 'Vintage Deep Dive' tab. Looking for all tabs...")
            all_tabs = await page.query_selector_all('[role="tab"], .tab, button.tab')
            if all_tabs:
                print(f"Found {len(all_tabs)} tab(s):")
                for i, tab in enumerate(all_tabs):
                    text = await tab.inner_text()
                    print(f"  Tab {i+1}: {text}")
                    if "Vintage Deep Dive" in text or "Deep Dive" in text:
                        print(f"  ✓ Clicking this tab!")
                        await tab.click()
                        tab_found = True
                        await asyncio.sleep(2)
                        break
        
        if tab_found:
            print("✓ Clicked 'Vintage Deep Dive' tab")
            
            # Take screenshot of the tab
            screenshot_path = screenshots_dir / "05_vintage_deep_dive_tab_top.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"✓ Screenshot saved: {screenshot_path}")
            
            # Scroll through this tab
            print("Scrolling through Vintage Deep Dive tab...")
            page_height = await page.evaluate('document.body.scrollHeight')
            scroll_positions = list(range(0, page_height, viewport_height // 2))
            
            for i, scroll_pos in enumerate(scroll_positions):
                await page.evaluate(f'window.scrollTo(0, {scroll_pos})')
                await asyncio.sleep(1)
                
                screenshot_path = screenshots_dir / f"05_vintage_deep_dive_scroll_{i+1:02d}_pos_{scroll_pos}.png"
                await page.screenshot(path=str(screenshot_path), full_page=False)
                print(f"✓ Screenshot saved: {screenshot_path} (scroll position: {scroll_pos}px)")
        else:
            print("⚠ Could not find 'Vintage Deep Dive' tab")
        
        print("\n" + "="*80)
        print("STEP 6: Extracting data from page...")
        print("="*80)
        
        # Scroll back to top
        await page.evaluate('window.scrollTo(0, 0)')
        await asyncio.sleep(1)
        
        # Extract all text content from the page
        page_text = await page.evaluate('''() => {
            return document.body.innerText;
        }''')
        
        # Save the text content
        text_file = screenshots_dir / "page_content.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(page_text)
        print(f"✓ Page text content saved to: {text_file}")
        
        # Try to extract specific data elements
        print("\nExtracting specific data elements...")
        
        # Look for data tables
        tables = await page.query_selector_all('table, .dash-table')
        print(f"Found {len(tables)} table(s)")
        
        for i, table in enumerate(tables):
            table_text = await table.inner_text()
            table_file = screenshots_dir / f"table_{i+1}_content.txt"
            with open(table_file, 'w', encoding='utf-8') as f:
                f.write(table_text)
            print(f"  ✓ Table {i+1} saved to: {table_file}")
        
        # Look for cards or metric displays
        cards = await page.query_selector_all('.card, [class*="metric"], [class*="kpi"]')
        print(f"Found {len(cards)} card/metric element(s)")
        
        for i, card in enumerate(cards[:20]):  # Limit to first 20
            try:
                card_text = await card.inner_text()
                if card_text.strip():
                    print(f"  Card/Metric {i+1}: {card_text[:100]}...")
            except:
                pass
        
        print("\n" + "="*80)
        print("✓ Automation complete!")
        print(f"✓ All screenshots saved to: {screenshots_dir}")
        print("="*80)
        
        # Keep browser open for a moment
        await asyncio.sleep(2)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
