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
from machine import WDT, reset

# ============================================================================
# CONFIGURATION — Edit these values before flashing
# ============================================================================

WIFI_SSID        = ""
WIFI_PASSWORD    = ""

DUCKDNS_TOKEN    = ""
DUCKDNS_DOMAIN   = ""
DUCKDNS_INTERVAL = 300  # Update every 5 minutes

# Timeouts and limits
WIFI_MAX_ATTEMPTS  = 15      # Max WiFi connection attempts before reboot
WIFI_RETRY_DELAY_S = 2       # Initial delay between WiFi retries (seconds)
WIFI_MAX_DELAY_S   = 30      # Max backoff delay for WiFi retries (seconds)
HTTP_TIMEOUT_S     = 15      # HTTP request timeout (seconds)
WDT_TIMEOUT_S      = 120     # Watchdog timeout (seconds)
LOOP_SLEEP_S       = 10      # Main loop sleep interval (seconds)

DUCKDNS_URL = "https://www.duckdns.org/update"

# ============================================================================
# WIFI
# ============================================================================

def wifi_connect(wdt):
    """Connect to WiFi with exponential backoff and max attempt limit.

    Raises RuntimeError if WIFI_MAX_ATTEMPTS is exceeded, so the caller
    (or the fatal handler) can reboot the device instead of hanging forever.
    """
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if wlan.isconnected():
        print("[WiFi] Already connected:", wlan.ifconfig()[0])
        return wlan

    print("[WiFi] Connecting to", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    attempts = 0
    delay = WIFI_RETRY_DELAY_S

    while not wlan.isconnected():
        attempts += 1
        if attempts > WIFI_MAX_ATTEMPTS:
            raise RuntimeError(
                "WiFi failed after %d attempts" % WIFI_MAX_ATTEMPTS
            )

        wdt.feed()
        print("[WiFi] Attempt %d/%d, waiting %ds..." % (
            attempts, WIFI_MAX_ATTEMPTS, delay))
        time.sleep(delay)
        delay = min(delay * 2, WIFI_MAX_DELAY_S)

        # Re-trigger connect in case the previous one timed out
        if not wlan.isconnected():
            try:
                wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            except Exception:
                pass

    print("[WiFi] Connected!", wlan.ifconfig()[0])
    return wlan


# ============================================================================
# DUCKDNS UPDATER
# ============================================================================

def update_duckdns(wdt):
    """Send current IP to DuckDNS. Returns True on success."""
    if not DUCKDNS_TOKEN or not DUCKDNS_DOMAIN:
        print("[DuckDNS] Not configured")
        return False

    url = "%s?domains=%s&token=%s&verbose=true" % (
        DUCKDNS_URL, DUCKDNS_DOMAIN, DUCKDNS_TOKEN)

    r = None
    try:
        wdt.feed()
        print("[DuckDNS] Updating %s.duckdns.org..." % DUCKDNS_DOMAIN)

        r = urequests.get(url, headers={"Connection": "close"},
                          timeout=HTTP_TIMEOUT_S)
        result = r.text.strip()
        status_code = r.status_code
        r.close()
        r = None
        gc.collect()

        if status_code != 200:
            print("[DuckDNS] HTTP error: %d" % status_code)
            return False

        # DuckDNS verbose response: first line is OK/KO
        # Remaining lines may contain IP and change info
        lines = result.split("\n")
        status = lines[0].strip() if lines else "UNKNOWN"
        ip     = lines[1].strip() if len(lines) > 1 else "N/A"
        change = lines[2].strip() if len(lines) > 2 else ""

        print("[DuckDNS] Status: %s" % status)
        print("[DuckDNS] IP: %s" % ip)
        if change:
            print("[DuckDNS] Change: %s" % change)

        if status == "KO":
            print("[DuckDNS] DuckDNS rejected the update")
            return False

        return status == "OK"

    except Exception as e:
        print("[DuckDNS] Error:", e)
        if r is not None:
            try:
                r.close()
            except Exception:
                pass
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
        print("[ERROR] DuckDNS not configured — halting.")
        return

    # Watchdog — reboot if stuck
    wdt = WDT(timeout=WDT_TIMEOUT_S * 1000)
    print("[WDT] Watchdog enabled (%ds)" % WDT_TIMEOUT_S)

    # Connect to WiFi (may raise on failure → handled by outer except)
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
            print(
                "\n[Status] Uptime: %dh %dm | Updates: %d OK, %d failed "
                "| RAM: %d KB free\n" % (
                    uptime // 3600,
                    (uptime % 3600) // 60,
                    success_count,
                    fail_count,
                    gc.mem_free() // 1024,
                )
            )

        time.sleep(LOOP_SLEEP_S)
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
