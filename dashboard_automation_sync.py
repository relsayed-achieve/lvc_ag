"""
Dashboard Automation Script (Synchronous version)
"""
import time
from playwright.sync_api import sync_playwright
from pathlib import Path

def main():
    # Create screenshots directory
    screenshots_dir = Path("dashboard_automation_screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    with sync_playwright() as p:
        # Launch browser
        print("Launching browser...")
        browser = p.chromium.launch(headless=True)
        
        print("Creating context...")
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        
        print("Creating page...")
        page = context.new_page()
        
        print("\nSTEP 1: Navigating to http://127.0.0.1:8051/")
        page.goto('http://127.0.0.1:8051/', wait_until='domcontentloaded', timeout=60000)
        
        # Wait for Dash to fully load
        print("Waiting for Dash to load...")
        try:
            page.wait_for_selector('.dash-dropdown, #vintage-week-selector, #overview-week-selector', timeout=15000)
        except:
            print("  Note: Dropdown selector not found, continuing anyway...")
        time.sleep(5)
        
        # Take initial screenshot
        screenshot_path = screenshots_dir / "01_initial_page.png"
        page.screenshot(path=str(screenshot_path))
        print(f"✓ Screenshot saved: {screenshot_path}")
        
        print("\nSTEP 2: Looking for week selector dropdown...")
        
        # Find the dropdown - try multiple selectors
        dropdown_found = False
        dropdown_selectors = [
            '#vintage-week-selector',
            '#overview-week-selector',
            '#inperiod-week-selector',
            'div[id*="week-selector"]',
            '.dash-dropdown'
        ]
        
        for selector in dropdown_selectors:
            try:
                dropdown = page.locator(selector).first
                if dropdown.count() > 0:
                    dropdown_found = True
                    print(f"✓ Found dropdown with selector: {selector}")
                    break
            except:
                continue
        
        if dropdown_found:
            try:
                # Click to open
                print("Clicking dropdown...")
                dropdown.click()
                time.sleep(2)
                
                # Take screenshot
                screenshot_path = screenshots_dir / "02_dropdown_opened.png"
                page.screenshot(path=str(screenshot_path))
                print(f"✓ Screenshot saved: {screenshot_path}")
                
                # Get all options - wait for them to appear
                print("Looking for options...")
                time.sleep(2)  # Give time for options to render
                
                # Try multiple ways to find the option
                option_found = False
                
                # Method 1: Try to click directly on text
                try:
                    option = page.get_by_text("2026-02-22", exact=True).first
                    if option.count() > 0:
                        print("✓ Found option '2026-02-22' by text")
                        option.click()
                        option_found = True
                except Exception as e:
                    print(f"  Method 1 failed: {e}")
                
                # Method 2: Try with different selectors
                if not option_found:
                    try:
                        options = page.locator('.Select-option, .VirtualizedSelectOption, div[role="option"], .Select-menu-outer > div > div').all()
                        print(f"Found {len(options)} option element(s)")
                        
                        for i, opt in enumerate(options):
                            try:
                                text = opt.inner_text().strip()
                                if text:  # Only print non-empty options
                                    print(f"  Option {i+1}: '{text}'")
                                if "02-22" in text or "2026-02-22" in text:
                                    print(f"  ✓ Clicking this option!")
                                    opt.click()
                                    option_found = True
                                    break
                            except Exception as e:
                                continue
                    except Exception as e:
                        print(f"  Method 2 failed: {e}")
                
                if option_found:
                    print("\nSTEP 3: Waiting for dashboard to refresh...")
                    time.sleep(8)  # Wait longer for data to load
                    
                    # Click "Expand All" button
                    try:
                        expand_btn = page.get_by_text("Expand All", exact=False).first
                        if expand_btn.count() > 0:
                            print("Clicking 'Expand All' button...")
                            expand_btn.click()
                            time.sleep(3)
                    except:
                        print("  Note: Could not find 'Expand All' button")
                    
                    screenshot_path = screenshots_dir / "03_after_week_selection.png"
                    page.screenshot(path=str(screenshot_path))
                    print(f"✓ Screenshot saved: {screenshot_path}")
                    
                    # Capture Overview tab content
                    print("Capturing Overview tab content...")
                    overview_text = page.evaluate('document.body.innerText')
                    overview_file = screenshots_dir / "overview_tab_content.txt"
                    with open(overview_file, 'w', encoding='utf-8') as f:
                        f.write(overview_text)
                    print(f"✓ Overview tab content saved to: {overview_file}")
                else:
                    print("⚠ Could not find 2026-02-22 option")
            except Exception as e:
                print(f"⚠ Error interacting with dropdown: {e}")
        else:
            print("⚠ Dropdown not found with any selector")
        
        print("\nSTEP 4: Scrolling through entire page...")
        
        # Get page height
        page_height = page.evaluate('document.body.scrollHeight')
        viewport_height = 1080
        scroll_step = viewport_height // 2
        
        scroll_pos = 0
        scroll_count = 1
        while scroll_pos < page_height:
            page.evaluate(f'window.scrollTo(0, {scroll_pos})')
            time.sleep(1)
            
            screenshot_path = screenshots_dir / f"04_scroll_{scroll_count:02d}_pos_{scroll_pos}.png"
            page.screenshot(path=str(screenshot_path))
            print(f"✓ Screenshot saved: {screenshot_path} (scroll: {scroll_pos}px)")
            
            scroll_pos += scroll_step
            scroll_count += 1
        
        # Scroll back to top
        page.evaluate('window.scrollTo(0, 0)')
        time.sleep(1)
        
        print("\nSTEP 5: Looking for 'Vintage Deep Dive' tab...")
        
        try:
            # Try to find the tab
            tab = page.get_by_text("Vintage Deep Dive", exact=False).first
            if tab.count() > 0:
                print("✓ Found 'Vintage Deep Dive' tab")
                tab.click()
                time.sleep(5)  # Wait for tab content to load
                
                # Change the week selector on this tab too
                print("Looking for week selector on Vintage Deep Dive tab...")
                try:
                    vintage_dropdown = page.locator('#vintage-week-selector').first
                    if vintage_dropdown.count() > 0:
                        print("✓ Found Vintage Deep Dive week selector")
                        vintage_dropdown.click()
                        time.sleep(2)
                        
                        # Click on 2026-02-22
                        option = page.get_by_text("2026-02-22", exact=True).first
                        if option.count() > 0:
                            print("✓ Selecting 2026-02-22 on Vintage Deep Dive tab")
                            option.click()
                            time.sleep(8)  # Wait for data to load
                except Exception as e:
                    print(f"  Error changing week on Vintage Deep Dive: {e}")
                
                # Click "Expand All" on this tab too
                try:
                    expand_btn = page.get_by_text("Expand All", exact=False).first
                    if expand_btn.count() > 0:
                        print("Clicking 'Expand All' on Vintage Deep Dive tab...")
                        expand_btn.click()
                        time.sleep(3)
                except:
                    print("  Note: Could not find 'Expand All' button on this tab")
                
                screenshot_path = screenshots_dir / "05_vintage_deep_dive_tab_top.png"
                page.screenshot(path=str(screenshot_path))
                print(f"✓ Screenshot saved: {screenshot_path}")
                
                # Scroll through this tab
                print("Scrolling through Vintage Deep Dive tab...")
                page_height = page.evaluate('document.body.scrollHeight')
                
                scroll_pos = 0
                scroll_count = 1
                while scroll_pos < page_height:
                    page.evaluate(f'window.scrollTo(0, {scroll_pos})')
                    time.sleep(1)
                    
                    screenshot_path = screenshots_dir / f"05_vintage_deep_dive_scroll_{scroll_count:02d}_pos_{scroll_pos}.png"
                    page.screenshot(path=str(screenshot_path))
                    print(f"✓ Screenshot saved: {screenshot_path} (scroll: {scroll_pos}px)")
                    
                    scroll_pos += scroll_step
                    scroll_count += 1
            else:
                print("⚠ 'Vintage Deep Dive' tab not found")
        except Exception as e:
            print(f"⚠ Error with tab: {e}")
        
        print("\n" + "="*80)
        print("STEP 6: Extracting data from page...")
        print("="*80)
        
        # Scroll back to top
        page.evaluate('window.scrollTo(0, 0)')
        time.sleep(1)
        
        # Extract all text from Vintage Deep Dive tab
        vintage_text = page.evaluate('document.body.innerText')
        
        text_file = screenshots_dir / "vintage_deepdive_tab_content.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(vintage_text)
        print(f"✓ Vintage Deep Dive tab content saved to: {text_file}")
        
        print("\n" + "="*80)
        print("✓ Automation complete!")
        print(f"✓ All screenshots saved to: {screenshots_dir}")
        print("="*80)
        
        # Keep browser open briefly
        time.sleep(3)
        
        browser.close()

if __name__ == "__main__":
    main()
