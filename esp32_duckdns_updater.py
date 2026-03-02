"""
ESP32-C3 Super Mini — DuckDNS Updater
======================================
Simple, reliable DuckDNS IP updater that runs periodically.

Hardware: ESP32-C3 Super Mini (400 KB SRAM, 4 MB flash)
Runtime:  MicroPython 1.20+

Updates DuckDNS every 5 minutes with current public IP.
"""

import network
import urequests
import time
import gc
from machine import WDT, reset, freq

# ============================================================================
# CONFIGURATION — Edit these values before flashing
# ============================================================================

WIFI_SSID        = ""
WIFI_PASSWORD    = ""

DUCKDNS_TOKEN    = ""
DUCKDNS_DOMAIN   = ""
DUCKDNS_INTERVAL = 300  # Update every 5 minutes

# ============================================================================
# WIFI
# ============================================================================

def wifi_connect(wdt=None):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        print("[WiFi] Already connected:", wlan.ifconfig()[0])
        return wlan
    print("[WiFi] Connecting to", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    delay = 2
    while not wlan.isconnected():
        if wdt: wdt.feed()  # Feed watchdog during long connection attempts
        print("[WiFi] Waiting %ds..." % delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)
        if not wlan.isconnected():
            try: wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            except Exception: pass
    print("[WiFi] Connected!", wlan.ifconfig()[0])
    return wlan

# ============================================================================
# DUCKDNS UPDATER
# ============================================================================

def update_duckdns(wdt=None):
    if not DUCKDNS_TOKEN or not DUCKDNS_DOMAIN:
        print("[DuckDNS] Not configured")
        return False
    
    r = None
    try:
        if wdt: wdt.feed()  # Feed watchdog before HTTP request
        print("[DuckDNS] Updating %s.duckdns.org..." % DUCKDNS_DOMAIN)
        r = urequests.get(
            "http://www.duckdns.org/update?domains=%s&token=%s&verbose=true" % (DUCKDNS_DOMAIN, DUCKDNS_TOKEN),
            headers={"Connection": "close"})
        result = r.text.strip()
        r.close(); r = None
        gc.collect()
        
        # Parse result
        lines = result.split("\n")
        status = lines[0] if lines else "UNKNOWN"
        ip = lines[1] if len(lines) > 1 else "N/A"
        change = lines[2] if len(lines) > 2 else ""
        
        print("[DuckDNS] Status: %s" % status)
        print("[DuckDNS] IP: %s" % ip)
        if change:
            print("[DuckDNS] Change: %s" % change)
        
        return status.startswith("OK")
    except Exception as e:
        print("[DuckDNS] Error:", e)
        if r:
            try: r.close()
            except: pass
        gc.collect()
        return False

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    boot_time = time.time()
    print("=" * 40)
    print("ESP32-C3 DuckDNS Updater starting...")
    print("Domain: %s.duckdns.org" % DUCKDNS_DOMAIN)
    print("Update interval: %d seconds" % DUCKDNS_INTERVAL)
    print("=" * 40)
    
    if not DUCKDNS_TOKEN or not DUCKDNS_DOMAIN:
        print("[ERROR] DuckDNS not configured!")
        return
    
    # Watchdog - reboot if stuck for 120 seconds
    wdt = WDT(timeout=120000)
    print("[WDT] Watchdog enabled (120s)")
    
    # Connect to WiFi
    wlan = wifi_connect(wdt)
    
    # First update
    update_duckdns(wdt)
    last_update = time.time()
    
    print("\n[Loop] Starting update loop...")
    success_count = 0
    fail_count = 0
    
    while True:
        wdt.feed()
        
        # Check WiFi connection
        if not wlan.isconnected():
            print("[WiFi] Lost connection, reconnecting...")
            wlan = wifi_connect(wdt)
            # Force immediate update after reconnect
            last_update = 0
        
        # Update if interval elapsed
        now = time.time()
        if now - last_update >= DUCKDNS_INTERVAL:
            if update_duckdns(wdt):
                success_count += 1
            else:
                fail_count += 1
            last_update = now
            
            # Status summary
            uptime = now - boot_time
            print("\n[Status] Uptime: %dh %dm | Updates: %d OK, %d failed | RAM: %d KB free\n" %
                  (uptime//3600, (uptime%3600)//60, success_count, fail_count, gc.mem_free()//1024))
        
        # Sleep 10 seconds between checks
        time.sleep(10)
        gc.collect()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Exit] Stopped by user")
    except Exception as e:
        print("\n[FATAL]", e)
        print("[FATAL] Rebooting in 10s...")
        time.sleep(10)
        reset()
