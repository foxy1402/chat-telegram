// ============================================================================
// EVN POWER-OUTAGE MONITOR (cskh.evnspc.vn) -> Telegram
// ============================================================================
// WHAT THIS DOES
//   Polls EVN SPC for scheduled outages ("lich ngung giam cung cap dien") for
//   each customer code below and messages Telegram ONLY when outages exist.
//   Built for a daily time-driven trigger: silence = nothing scheduled.
//
// FINDINGS ABOUT THE ENDPOINT (verified 08/08/2026 by live probing)
//   1. GET returns an ASP.NET-MVC HTML PARTIAL VIEW - never JSON.
//   2. HTTP status is ALWAYS 200, even for a nonexistent maKH (it just renders
//      the "no outage" template). Never branch on status codes for data.
//   3. Two query modes:
//        ?maKH=<code>&ChucNang=MaKhachHang      (per customer - this script)
//        ?madvi=<6-char>&ChucNang=MaDonVi       (per "dien luc" unit)
//      Company codes: PB01 Dong Nai, PB03 Lam Dong, PB05 Tay Ninh,
//      PB07 Dong Thap, PB10 Vinh Long, PB11 Can Tho, PB12 An Giang, PB14 Ca Mau.
//      Unit list:   /TraCuu/GetDanhMucDienLuc?pMA_DVICTREN=<company>
//      maKH -> madvi mapping = maKH prefix: PKdummy => madvi PK0200
//      (verified: PK0200 = Dien luc Dau Giay, Dong Nai). A bare company code
//      (e.g. madvi=PB11) is NOT accepted - always use 6-char unit codes.
//   4. Dates must be dd-MM-yyyy. If tuNgay/denNgay are OMITTED the server still
//      returns real data (response is byte-identical) using its own internal
//      default range - only the echoed date labels render blank. Always pass
//      both explicitly.
//   5. The portal itself queries a 5-day window (today -> today+4), mirrored
//      below via WINDOW_DAYS = 4.
//
// REAL RESPONSE TEMPLATES (match against these - do not invent fields)
//   No outage (<small class="red">, note LOWERCASE "khong"):
//     Hiện tại khách hàng <small class="red">không có lịch ngừng giảm cung cấp điện</small>
//     (unit mode reads "Hiện tại Điện lực" instead)
//   Outage(s) (<span class="red">):
//     Thông báo lịch ngừng giảm cung cấp điện
//     <span><b>Đơn vị:</b> Điện lực ...</span>
//     then one <div class="entry"> per outage:
//       <b>KHU VỰC:</b> Một phần đường Nguyễn Văn Cừ P Cái Khế TPCT
//       <b>THỜI GIAN:</b> Từ 07:30:00 ngày 11/08/2026 đến 11:00:00 ngày 11/08/2026
//       <b>LÝ DO NGỪNG CUNG CẤP ĐIỆN:</b> <span>Bảo trì, sửa chữa lưới điện</span>
//
//   There is NO "MÃ LỊCH" field anywhere. An earlier version detected the
//   outage banner with the wrong string ("Thông báo lịch cắt điện") and keyed
//   parsing on "MÃ LỊCH" - so real outages were silently reported as
//   "no outage". Detection below matches the exact lowercased markers.
//   Vietnamese text arrives hex-entity-encoded (&#x1ED9;...) -> decode first.
//
// TESTING TIP FOR FUTURE AI
//   Customer codes rarely have outages, so unit queries return instant data:
//     ...?madvi=PB1101&tuNgay=...&denNgay=...&ChucNang=MaDonVi   (Ninh Kieu)
//     ...?madvi=PB0101&...                                       (Dong Xoai)
//   Manual UI: https://www.cskh.evnspc.vn/TraCuu/LichNgungGiamCungCapDien
// ============================================================================

const CFG = {
  BOT_TOKEN: "dummy",
  CHAT_ID: "dummy",
  CUSTOMER_CODES: ["PKdummy"],
  WINDOW_DAYS: 4,       // official portal window: today -> today+4
  MAX_RETRIES: 3,
  RETRY_DELAY_MS: 2000,
  SEND_ERRORS: false,   // true = also notify Telegram on fetch/parse failures
  API_URL: "https://www.cskh.evnspc.vn/TraCuu/GetThongTinLichNgungGiamCungCapDien",
};

// Exact markers from the verified templates (compared on lowercased HTML)
const MARK_NO_OUTAGE = 'không có lịch ngừng giảm cung cấp điện';
const MARK_HAS_OUTAGE = 'thông báo lịch ngừng giảm cung cấp điện';

function evn() {
  const tz = 'Asia/Ho_Chi_Minh';
  const now = new Date();
  const fromDate = Utilities.formatDate(now, tz, 'dd-MM-yyyy');
  const toDate = Utilities.formatDate(new Date(now.getTime() + CFG.WINDOW_DAYS * 864e5), tz, 'dd-MM-yyyy');

  const alerts = [];
  const errors = [];

  CFG.CUSTOMER_CODES.forEach((code) => {
    const url = `${CFG.API_URL}?tuNgay=${fromDate}&denNgay=${toDate}&maKH=${code}&ChucNang=MaKhachHang`;
    const html = fetchWithRetry_(url);

    if (!html.ok) {
      errors.push(`${code}: ${html.error}`);
      return;
    }

    const lower = html.text.toLowerCase();
    if (lower.includes(MARK_HAS_OUTAGE)) {
      alerts.push({ code, outages: parseEvnHtml_(html.text) });
    } else if (!lower.includes(MARK_NO_OUTAGE)) {
      // Neither marker = template changed upstream; log for debugging
      errors.push(`${code}: unrecognized response`);
      console.log(`Unrecognized response for ${code}:\n${html.text.substring(0, 500)}`);
    }
    // MARK_NO_OUTAGE => intentionally silent
  });

  console.log(`Done: ${alerts.length} alert(s), ${errors.length} error(s)`);

  if (alerts.length) {
    sendToTelegram_(buildMessage_(alerts, fromDate, toDate));
  }
  if (errors.length) {
    console.error(errors.join('\n'));
    if (CFG.SEND_ERRORS) sendToTelegram_(`⚠️ EVN monitor lỗi:\n${errors.join('\n')}`);
  }
}

function fetchWithRetry_(url) {
  let lastError = null;
  for (let attempt = 1; attempt <= CFG.MAX_RETRIES; attempt++) {
    try {
      if (attempt > 1) Utilities.sleep(CFG.RETRY_DELAY_MS);
      const res = UrlFetchApp.fetch(url, {
        method: 'get',
        muteHttpExceptions: true, // endpoint always returns 200; still guard transport
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
          'Accept-Language': 'vi-VN,vi;q=0.9',
        },
      });
      const status = res.getResponseCode();
      if (status === 200) return { ok: true, text: res.getContentText() };
      lastError = `HTTP ${status}`;
      if (status < 500 && status !== 429) break; // 4xx won't fix itself by retrying
    } catch (e) {
      lastError = e.message;
    }
  }
  return { ok: false, error: lastError };
}

function parseEvnHtml_(html) {
  const outages = [];
  html = decodeHtmlEntities_(html);

  // One <div class="entry"> per outage; index 0 is the header before the first entry
  html.split(/<div\s+class="entry">/i).slice(1).forEach((entry) => {
    const field = (re) => {
      const m = entry.match(re);
      return m ? m[1].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim() : null;
    };

    const khuVuc = field(/<b>\s*KHU VỰC:\s*<\/b>([\s\S]*?)<\/span>/i);
    const lyDo = field(/LÝ DO NGỪNG CUNG CẤP ĐIỆN:<\/b>\s*<span>([\s\S]*?)<\/span>/i);

    // Captures HH:MM; seconds (always ":00" from EVN) are optional and dropped
    const t = entry.match(/Từ\s+(\d{1,2}:\d{2})(?::\d{2})?\s+ngày\s+([\d\/]+)\s+đến\s+(\d{1,2}:\d{2})(?::\d{2})?\s+ngày\s+([\d\/]+)/i);
    const thoiGian = t ? `${t[1]} ${t[2]} → ${t[3]} ${t[4]}` : null;

    if (khuVuc || thoiGian || lyDo) outages.push({ khuVuc, thoiGian, lyDo });
  });

  // Banner said "has outage" but entries couldn't be parsed (template drift):
  // still alert - a missed outage is worse than a vague one.
  if (!outages.length) {
    outages.push({
      khuVuc: null,
      thoiGian: 'Xem chi tiết tại cskh.evnspc.vn',
      lyDo: 'Có thông báo cúp điện nhưng không đọc được chi tiết',
    });
  }
  console.log(`Parsed ${outages.length} outage(s)`);
  return outages;
}

function buildMessage_(alerts, fromDate, toDate) {
  const lines = [`🔌 CẢNH BÁO CÚP ĐIỆN  (${fromDate} → ${toDate})`];
  alerts.forEach(({ code, outages }) => {
    lines.push('', `📋 Mã KH: ${code} — ${outages.length} lịch cúp điện`);
    outages.forEach((o, i) => {
      lines.push(`━━━━ ⚡ Lịch #${i + 1}`);
      if (o.khuVuc) lines.push(`📍 ${o.khuVuc}`);
      if (o.thoiGian) lines.push(`⏰ ${o.thoiGian}`);
      if (o.lyDo) lines.push(`📝 ${o.lyDo}`);
    });
  });
  // Sent as plain text (no parse_mode): addresses can contain _ or * which
  // would break Telegram Markdown parsing and drop the whole alert.
  return lines.join('\n').substring(0, 4000); // Telegram limit is 4096 chars
}

function decodeHtmlEntities_(text) {
  return text
    .replace(/&#x([0-9A-Fa-f]+);/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCharCode(parseInt(dec, 10)));
}

function sendToTelegram_(text) {
  const res = UrlFetchApp.fetch(`https://api.telegram.org/bot${CFG.BOT_TOKEN}/sendMessage`, {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify({ chat_id: CFG.CHAT_ID, text }),
    muteHttpExceptions: true,
  });
  const result = JSON.parse(res.getContentText());
  if (!result.ok) console.error('Telegram API error:', result, '\nMessage was:', text);
  else console.log('Telegram message sent');
}
