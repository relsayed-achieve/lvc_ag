#!/usr/bin/env python3
"""
Patient capture with longer waits for Dash callbacks
"""
from playwright.sync_api import sync_playwright
import time
import os

def main():
    screenshots_dir = "dashboard_patient_capture"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1200},
            device_scale_factor=1
        )
        page = context.new_page()
        
        print("Loading dashboard...")
        page.goto('http://127.0.0.1:8051/', timeout=30000)
        print("Waiting 10 seconds for initial load...")
        time.sleep(10)
        
        # TAB 1: Overview
        print(f"\n{'='*70}")
        print("TAB 1: OVERVIEW")
        print(f"{'='*70}")
        
        # Click Expand All
        try:
            page.click('button:has-text("Expand All")')
            print("Clicked Expand All, waiting 10 seconds...")
            time.sleep(10)
        except:
            print("Could not click Expand All")
        
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)
        
        page_height = page.evaluate("document.body.scrollHeight")
        print(f"Page height: {page_height}px")
        
        page.screenshot(path=f"{screenshots_dir}/TAB1_OVERVIEW_full.png", full_page=True)
        print("✓ Full page screenshot")
        
        # Scroll captures
        for i, pos in enumerate([0, 800, 1600, 2400]):
            if pos < page_height:
                page.evaluate(f"window.scrollTo(0, {pos})")
                time.sleep(2)
                page.screenshot(path=f"{screenshots_dir}/TAB1_OVERVIEW_at_{pos}px.png")
                print(f"✓ Screenshot at {pos}px")
        
        text = page.evaluate("document.body.innerText")
        with open(f"{screenshots_dir}/TAB1_OVERVIEW_text.txt", "w") as f:
            f.write(text)
        
        # TAB 2: Vintage Deep Dive
        print(f"\n{'='*70}")
        print("TAB 2: VINTAGE DEEP DIVE")
        print(f"{'='*70}")
        
        page.click('text="Vintage Deep Dive"')
        print("Clicked tab, waiting 15 seconds for callbacks...")
        time.sleep(15)
        
        # Click Expand All
        try:
            page.click('button:has-text("Expand All")')
            print("Clicked Expand All, waiting 15 seconds...")
            time.sleep(15)
        except:
            print("Could not click Expand All")
        
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)
        
        page_height = page.evaluate("document.body.scrollHeight")
        print(f"Page height: {page_height}px")
        
        page.screenshot(path=f"{screenshots_dir}/TAB2_VINTAGE_DEEP_DIVE_full.png", full_page=True)
        print("✓ Full page screenshot")
        
        for i, pos in enumerate([0, 800, 1600, 2400, 3200, 4000]):
            if pos < page_height:
                page.evaluate(f"window.scrollTo(0, {pos})")
                time.sleep(2)
                page.screenshot(path=f"{screenshots_dir}/TAB2_VINTAGE_DEEP_DIVE_at_{pos}px.png")
                print(f"✓ Screenshot at {pos}px")
        
        text = page.evaluate("document.body.innerText")
        with open(f"{screenshots_dir}/TAB2_VINTAGE_DEEP_DIVE_text.txt", "w") as f:
            f.write(text)
        
        # TAB 3: In-Period Deep Dive
        print(f"\n{'='*70}")
        print("TAB 3: IN-PERIOD DEEP DIVE")
        print(f"{'='*70}")
        
        page.click('text="In-Period Deep Dive"')
        print("Clicked tab, waiting 15 seconds for callbacks...")
        time.sleep(15)
        
        # Click Expand All
        try:
            page.click('button:has-text("Expand All")')
            print("Clicked Expand All, waiting 15 seconds...")
            time.sleep(15)
        except:
            print("Could not click Expand All")
        
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)
        
        page_height = page.evaluate("document.body.scrollHeight")
        print(f"Page height: {page_height}px")
        
        page.screenshot(path=f"{screenshots_dir}/TAB3_IN_PERIOD_DEEP_DIVE_full.png", full_page=True)
        print("✓ Full page screenshot")
        
        for i, pos in enumerate([0, 800, 1600, 2400, 3200, 4000]):
            if pos < page_height:
                page.evaluate(f"window.scrollTo(0, {pos})")
                time.sleep(2)
                page.screenshot(path=f"{screenshots_dir}/TAB3_IN_PERIOD_DEEP_DIVE_at_{pos}px.png")
                print(f"✓ Screenshot at {pos}px")
        
        text = page.evaluate("document.body.innerText")
        with open(f"{screenshots_dir}/TAB3_IN_PERIOD_DEEP_DIVE_text.txt", "w") as f:
            f.write(text)
        
        browser.close()
        
        print(f"\n{'='*70}")
        print("COMPLETE!")
        print(f"{'='*70}")
        print(f"Location: {os.path.abspath(screenshots_dir)}")
        
        files = sorted(os.listdir(screenshots_dir))
        print(f"\nTotal: {len(files)} files")
        for f in files:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
