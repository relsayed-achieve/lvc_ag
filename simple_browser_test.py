"""Simple browser test to check if Playwright works"""
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    print("Launching browser...")
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    print("Navigating to dashboard...")
    page.goto('http://127.0.0.1:8051/', wait_until='domcontentloaded')
    
    print("Waiting for content...")
    time.sleep(5)
    
    print("Getting page title...")
    title = page.title()
    print(f"Page title: {title}")
    
    print("Getting page content...")
    content = page.content()
    print(f"Page content length: {len(content)} characters")
    
    print("Taking screenshot...")
    page.screenshot(path='test_screenshot.png')
    print("Screenshot saved!")
    
    browser.close()
    print("Done!")
