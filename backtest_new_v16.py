"""
backtest_new_v16.py - 台股月營收動能策略回測 new_v16
（修正 0050 除息還原 ＋ 0050 1拆4分割保護 ＋ 漲停鎖死防呆 ＋ 極端基期 YoY 上限 ＋ 減資異常保護）

相較 backtest_new_v4.py 的新增升級：
  1. 【新增「當月 YoY 必須大於 Min_YoY」的底線要求（Config / CLI / Grid 皆可調）】：
       * `Min_YoY`（預設 0.0%，即要求當月營收至少要是年增 `YoY > 0%`；可設如 10.0、20.0；設 None 或 <= -999 代表關閉不設限）：
       * 在 v4 中，若某公司上月 YoY 為 -55%、本月 YoY 為 -24%，雖然 `YoY_Diff = +31% > 30%`，但本質上仍是營收年減 24% 的衰退股（如 6283 淳安）。
       * 加入 `Min_YoY` 底線後，進場訊號必須同時滿足「`當月 YoY > Min_YoY`」且「`(YoY月差值 > X_YoY_Diff) 或 (連3月MoM月差值 > Y_MoM_Diff)`」，
         確保選出的是「真正處於正成長（或高成長）且正在加速」的標的。
  2. 【完整保留 v4 其餘所有功能與設定】：
       * 成長加速度門檻：`X_YoY_Diff`（當月YoY - 上月YoY > X）、`Y_MoM_Diff`（連3月 當月MoM - 上月MoM > Y）
       * 三時段可調進場時機（`Pre_Market_Entry`、`Intraday_Entry`、`Post_Market_Entry`）
       * 盤中觸價停損（`Stop_Loss_Mode = 'intraday'`）與移動停利（`Trail_Stop_Pct`）
       * 買進時點漲跌幅確認（`Min_Buy_Day_Chg`）與早鳥公告過濾（`Min_Ann_Day`）

用法（在本資料夾）：
  python backtest_new_v16.py --single                         # 單次回測（依檔案內 PARAMS 設定）
  python backtest_new_v16.py --single --min-yoy 20            # 要求當月 YoY > 20% 且滿足成長差值門檻
  python backtest_new_v16.py --single --min-yoy -999          # 關閉當月 YoY 底線（100% 還原 new_v4 結果）
  python backtest_new_v16.py                                  # 執行 Grid 參數掃描（支援 --grid-min-yoy）
  python backtest_new_v16.py --download                       # 下載新日期行情，並將舊 CSV 升級補齊 high/low
"""
import argparse
import datetime as dt
import glob
import io
import itertools
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 可調參數（沿用使用者於 v4 的最新設定 + 新增 Min_YoY 底線參數）
# =====================================================================
PARAMS = {
    # --- v4/v5 進場訊號（成長加速度門檻 + v5 當月 YoY 底線）---
    "X_YoY_Diff": 10.0,              # 單月 YoY 對比上月 YoY 差值（YoY_t - YoY_{t-1}）> X %
    "Y_MoM_Diff": 2.0,               # 或：連續 3 個月（t, t-1, t-2）MoM 對比上月 MoM 差值（MoM_k - MoM_{k-1}）皆 > Y %
    "Min_YoY": 0.0,                  # v5 新增：當月 YoY 必須 > 此底線(%) 才可買入（例如 0=要求正成長、20=要求YoY>20%；設 <= -999 代表關閉不設限）
    "Max_YoY": 500.0,                # v14 新增：當月 YoY 上限(%)，過濾低基期或一次性入帳之極端異常股（設 >= 99999 代表關閉）

    # --- 三時段進場時機設定（可調 Config）---
    "Pre_Market_End": "09:00:00",    # 盤前公告分界時間（< 09:00:00 視為盤前公告）
    "Cutoff_Time": "13:25:00",       # 盤中/盤後公告分界時間（09:00:00 ~ 13:25:00 為盤中；> 13:25:00 為盤後）
    "Pre_Market_Entry": "open",      # 盤前公告買進時機：'open'=當天開盤即買入 | 'close'=當天收盤買入 | 'skip'=不買
    "Intraday_Entry": "close",       # 盤中公告買進時機：'immediate'=公告當下即買進 | 'close'=當日收盤買入 | 'next_open'=隔日開盤買 | 'skip'=不買
    "Intraday_Price_Proxy": "mid",   # 盤中「當下買進」之日K價格代理：'mid'=(open+close)/2 | 'hl2'=(high+low)/2 | 'close'=收盤價 | 'open'=開盤價
    "Post_Market_Entry": "next_open",# 盤後公告買進時機：'next_open'=隔天開盤即買入 | 'next_close'=隔天收盤買入 | 'skip'=不買

    # --- 公告反應日價格確認 & 早鳥公告過濾 ---
    "Min_Buy_Day_Chg": -50.0,          # 買進時點漲跌幅須 >= 此值(%) 才買（開盤買看開盤漲跌 gap_pct；收盤買看收盤漲跌 up_pct；<=-50 代表關閉）
    "Max_Buy_Day_Chg": 50.0,            # 防追高1：買進時點當日漲跌幅須 <= 此值(%) 才買（開盤買看gap_pct，收盤買看up_pct；>=50 代表關閉）
    "Pre_Runup_Days": 3,             # 防追高2：檢查進場前 N 個交易日的波段累積漲幅（預設 3 日）
    "Max_Pre_Runup_Pct": 999.0,       # 防追高2：進場價相對 N 日前收盤價累積漲幅須 <= 此值(%) 才買（>=999 代表關閉）
    "Min_Ann_Day": 1,                # 營收公告日（當月幾號）須 >= 此值才買（1 = 不過濾；設 6 代表排除每月 1~5 日早鳥公告）

    # --- 出場 ---
    "N_Days": 16,                    # 持有 N 個交易日（買進日 + N 個交易日開盤賣出）
    "Stop_Loss_Pct": 9.0,           # 停損 X%（0 = 不停損）
    "Stop_Loss_Mode": "intraday",    # 'intraday' = 盤中最低價(low)跌破 X% 當日即停損；'close' = 收盤跌破 X% 隔日開盤賣出
    "Dyn_SL_Cap_Pct": -999.0,          # 動態上移停損上限 X%（每漲1%停損上移1%，最多拉高至此%；例 0=拉到保本、5=拉到鎖利+5%；設 <= -999 關閉）
    "Trail_Stop_Pct": 0.0,           # 賺錢後，收盤價自持有期間最高價回落 >= Y% → 隔天開盤賣出；0 = 關閉
    "Trail_Activate_Pct": 0.0,       # 移動停利啟動門檻（最高價 > 買進價 × (1+此%) 才啟動）

    # --- v16 新增：季報 EPS 基本面濾網（嚴格依財報公告截止日防未來函數）---
    "Enable_EPS_Filter": True,       # 是否啟用季報 EPS 濾網（若無 EPS 快取會自動提示並略過）
    "Min_Last_Q_EPS": -1.0,           # 最新已公告單季 EPS 須 > 此值（例 0.0=單季賺錢；設 <= -999 關閉）
    "Min_TTM_EPS": -999.0,           # 近四季累計 EPS (TTM) 須 > 此值（例 0.0=近一年獲利；設 <= -999 關閉）
    "Min_Q_EPS_YoY": -999.0,         # 最新單季 EPS 年增率(%) 須 > 此值（例 0.0=單季獲利年增；設 <= -999 關閉）
    "Max_PE_Ratio": 999.0,           # 本益比上限（股價 / 近四季TTM EPS 須 <= 此值；設 >= 999 關閉）

    # --- 流動性 / 規模濾網 ---
    "Size_Mode": "adj_cap",          # 規模分位基準：\'adj_cap\'=還原歷史股本市值(消除配股/減資偏差) | \'cap\'=靜態股本市值 | \'turnover\'=均成交額
    "Cap_Decile_Low": 1,             # 市值十分位下界（含）
    "Cap_Decile_High": 4,            # 市值十分位上界（含）
    "Min_Avg_Volume": 50,           # 近 Vol_Window 日均量下限（張）
    "Min_Avg_Turnover": 0,                   # 固定均成交金額保底下限（元，設 0 代表交由動態比例控制）
    "Max_Turnover_Pct": 0.03,                # 單筆投入預算占個股20日均成交額上限比例（0.03 = 3%，即均成交額須 >= 投入金額 / 0.03；設 0 關閉）
    # --- v11 新增：空倉閒置資金自動持有 0050（獨立 SMA 進場 + 獨立停損）---
    "Enable_Idle_ETF": False,                 # 是否開啟閒置資金自動持有 ETF
    "Idle_ETF_Code": "0050",                 # 停泊 ETF 代號（預設 0050）
    "ETF_SMA_Days": 60,                      # ETF 進場均線天數（昨收 > SMA 且 SMA 向上才買進；0=不限均線）
    "ETF_Stop_Loss_Pct": 2.0,                # ETF 獨立停損 %（跌破平均成本此比例即停損；0=不停損）
    "ETF_Cash_Buffer_Pct": 5.0,              # 保留現金緩衝 %（避免頻繁零碎買賣 ETF）
    "Vol_Window": 20,

    # --- 資金控管 ---
    "Position_Pct": 4,             # 功能1：每檔投入當下動態總資產的 X%（設 0 改用固定金額）
    "Position_Size": 50_000,     # 備用固定金額上限
    "Enable_Gap_Short": False,        # 功能2：開盤跳空過熱放空 + 收盤反手作多
    "Gap_Short_Min_Pct": 8.0,        # 開盤跳空漲幅 > X% 觸發放空
    "Gap_Short_Max_Pct": 10.0,        # 開盤跳空漲幅 < Y% 觸發放空
    "Short_Stop_Loss_Pct": 2.5,      # 放空盤中停損 %（0=不停損）
    "Reverse_Check_Buy_Chg": False,  # 反手作多時是否也要檢查 Min_Buy_Day_Chg（預設 False 不受限）
    "Max_Positions": 25,             # 同時最多持有檔數

    # --- 成交可行性 ---
    "Limit_Pct": 9.5,                # 買進時點漲幅 >= 此值視為鎖漲停買不到；賣出時開盤跌幅 <= -此值視為賣不掉
}

BT = {
    "Initial_Capital": 3_000_000,
    "Fee_Discount": 0.28,            # 手續費折扣（2.8 折）
    "Slippage": 0.0015,              # 單邊滑價
    "Tax": 0.003,                    # 一般證交稅 0.3%
    "DayTrade_Tax": 0.0015,          # 現股當沖證交稅 0.15%
}
RF_RATE = 1.5                        # 無風險利率（年化 %），Martin Ratio 用

EXCLUDED_KEYWORDS = ["營造", "建材", "生技", "生物科技", "金融", "保險"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
PRICE_DIR = os.path.join(CACHE_DIR, "prices")
OLD_REV_DIR = os.path.join(CACHE_DIR, "revenue")
EPS_DIR = os.path.join(CACHE_DIR, "eps")
EXT_REV_DIR = OLD_REV_DIR  # 直接指向整合後的 data_cache/revenue
OUT_ROOT = os.path.join(BASE_DIR, "backtest_output")
for _d in (PRICE_DIR, OLD_REV_DIR, EPS_DIR, OUT_ROOT):
    os.makedirs(_d, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}
PRICE_COLS = ["market", "stock_id", "volume", "open", "high", "low", "close", "chg"]


# =====================================================================
# 一、資料層（每日行情含 High/Low / 精確申報時間月營收 / 產業別 / 發行股數）
# =====================================================================
def is_excluded_industry(industry) -> bool:
    s = str(industry or "")
    return any(k in s for k in EXCLUDED_KEYWORDS)


def ym_str(y, m):
    return f"{y}-{m:02d}"


def add_months(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1


def _get(url, params=None, tries=4, wait=5):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(wait * (i + 1))
    raise last


def _num(x):
    if x is None:
        return np.nan
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return np.nan


# ---------------- 每日行情（含 open, high, low, close）----------------
def _fetch_twse_day(d):
    j = _get("https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
             params={"date": d.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"}).json()
    stat = str(j.get("stat", ""))
    if stat != "OK":
        if "沒有符合" in stat:
            return pd.DataFrame(columns=PRICE_COLS)
        raise RuntimeError(f"TWSE 回應異常: {stat}")
    tbl = next((t for t in j.get("tables", [])
                if "證券代號" in (t.get("fields") or []) and "開盤價" in (t.get("fields") or []) and t.get("data")), None)
    if tbl is None:
        return pd.DataFrame(columns=PRICE_COLS)
    rows = []
    for r in tbl["data"]:
        code = str(r[0]).strip()
        if not re.fullmatch(r"\d{4}", code):
            continue
        mm = re.search(r">\s*([+\-xX])", str(r[9]))
        sg = mm.group(1) if mm else str(r[9]).strip()
        sign = 1.0 if sg == "+" else (-1.0 if sg == "-" else 0.0)
        diff = _num(r[10])
        chg = sign * diff if not np.isnan(diff) else np.nan
        if sg in ("x", "X"):
            chg = np.nan
        rows.append(("L", code, _num(r[2]), _num(r[5]), _num(r[6]), _num(r[7]), _num(r[8]), chg))
    return pd.DataFrame(rows, columns=PRICE_COLS)


def _fetch_tpex_day(d):
    j = _get("https://www.tpex.org.tw/www/zh-tw/afterTrading/otc",
             params={"date": d.strftime("%Y/%m/%d"), "type": "EW", "response": "json"}).json()
    tables = j.get("tables") or []
    if not tables or not tables[0].get("data"):
        return pd.DataFrame(columns=PRICE_COLS)
    rows = []
    for r in tables[0]["data"]:
        code = str(r[0]).strip()
        if re.fullmatch(r"\d{4}", code):
            rows.append(("O", code, _num(r[7]), _num(r[4]), _num(r[5]), _num(r[6]), _num(r[2]), _num(str(r[3]).replace("+", ""))))
    return pd.DataFrame(rows, columns=PRICE_COLS)


def _price_path(d):
    return os.path.join(PRICE_DIR, d.strftime("%Y%m%d") + ".csv")


def _day_is_final(d):
    now = dt.datetime.now()
    if d > now.date():
        return False
    return not (d == now.date() and now.hour * 60 + now.minute < 15 * 60)


def _csv_has_hl(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split(",")
        return ("high" in header) and ("low" in header)
    except Exception:  # noqa: BLE001
        return False


def update_price_cache(start, end, sleep_sec=3.5):
    """下載缺少的平日行情，並自動將尚未包含 high/low 欄位的舊 CSV 重新抓取升級（支援中斷續傳）。"""
    missing_days, upgrade_days, d = [], [], start
    while d <= end:
        if d.weekday() < 5 and _day_is_final(d):
            p = _price_path(d)
            if not os.path.exists(p):
                missing_days.append(d)
            elif not _csv_has_hl(p):
                upgrade_days.append(d)
        d += dt.timedelta(days=1)

    days = missing_days + upgrade_days
    if days:
        print(f"📥 需下載/升級 {len(days)} 個平日行情（新日期 {len(missing_days)} 天 + 補齊 high/low 欄位 {len(upgrade_days)} 天），"
              f"預估約 {len(days) * sleep_sec * 2 / 60:.0f} 分鐘（可隨時 Ctrl+C 中斷續傳）")
    else:
        print("✅ 所有平日行情 CSV 皆已存在且已包含 high/low 欄位，無需下載。")
        return

    for i, d in enumerate(days, 1):
        try:
            twse = _fetch_twse_day(d)
            time.sleep(sleep_sec)
            if twse.empty:
                pd.DataFrame(columns=PRICE_COLS).to_csv(_price_path(d), index=False)
            else:
                tpex = _fetch_tpex_day(d)
                time.sleep(sleep_sec)
                if tpex.empty:
                    raise RuntimeError("上櫃資料為空（疑似限流），不寫入快取")
                pd.concat([twse, tpex], ignore_index=True).to_csv(_price_path(d), index=False)
            if i % 20 == 0 or i == len(days):
                print(f"   行情進度 {i}/{len(days)}  ({d})")
        except KeyboardInterrupt:
            print(f"\n⏹️ 使用者中斷下載（已完成 {i - 1}/{len(days)} 天，下次加 --download 會自動從中斷處繼續）")
            break
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ {d} 下載失敗：{e}；等 30 秒後繼續（下次重跑會補抓）")
            time.sleep(30)


def ensure_etf_prices(prices: pd.DataFrame, etf_code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """確保行情表內包含完整的 ETF（如 0050）日K資料，若快取不足則自動向證交所抓取並快取。"""
    has_etf = (not prices.empty) and (prices["stock_id"] == etf_code).sum() > 200
    if has_etf:
        return prices
    etf_cache = os.path.join(CACHE_DIR, f"{etf_code}_prices.csv")
    if os.path.exists(etf_cache):
        edf = pd.read_csv(etf_cache, dtype={"stock_id": str})
        edf["date"] = pd.to_datetime(edf["date"])
        edf = edf[(edf["date"] >= pd.Timestamp(start)) & (edf["date"] <= pd.Timestamp(end))]
        if len(edf) > 200:
            prices = prices[prices["stock_id"] != etf_code]
            return pd.concat([prices, edf], ignore_index=True)
    print(f"📥 正在從證交所補抓 {etf_code} 歷史日行情（{start} ~ {end}）...")
    rows = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        try:
            j = _get("https://www.twse.com.tw/exchangeReport/STOCK_DAY",
                     params={"response": "json", "date": f"{y}{m:02d}01", "stockNo": etf_code}).json()
            if j.get("stat") == "OK" and j.get("data"):
                for r in j["data"]:
                    parts = str(r[0]).strip().split("/")
                    if len(parts) == 3:
                        d_str = f"{int(parts[0]) + 1911}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                        raw_chg = str(r[7]).strip()
                        chg_val = np.nan if ("X" in raw_chg or "x" in raw_chg) else _num(raw_chg.replace("+", ""))
                        rows.append(("L", etf_code, _num(r[1]), _num(r[3]), _num(r[4]), _num(r[5]), _num(r[6]), chg_val, d_str))
            time.sleep(2.5)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ {etf_code} {y}-{m:02d} 抓取失敗: {e}")
        y, m = add_months(y, m, 1)
    if rows:
        edf = pd.DataFrame(rows, columns=PRICE_COLS + ["date"])
        edf.to_csv(etf_cache, index=False)
        edf["date"] = pd.to_datetime(edf["date"])
        edf = edf[(edf["date"] >= pd.Timestamp(start)) & (edf["date"] <= pd.Timestamp(end))]
        prices = prices[prices["stock_id"] != etf_code]
        print(f"✅ {etf_code} 補抓完成，共 {len(edf)} 個交易日。")
        return pd.concat([prices, edf], ignore_index=True)
    return prices


def load_prices(start, end):
    frames, missing, no_hl_days, d = [], [], 0, start
    while d <= end:
        if d.weekday() < 5:
            p = _price_path(d)
            if os.path.exists(p):
                df = pd.read_csv(p, dtype={"stock_id": str})
                if not df.empty:
                    if "high" not in df.columns:
                        df["high"] = np.nan
                        no_hl_days += 1
                    if "low" not in df.columns:
                        df["low"] = np.nan
                    df["date"] = pd.Timestamp(d)
                    frames.append(df)
            elif _day_is_final(d):
                missing.append(d)
        d += dt.timedelta(days=1)
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLS + ["date"])
    return prices, missing, no_hl_days


# ---------------- 產業別對照表與發行股數 ----------------
def load_industry_map():
    cache_path = os.path.join(CACHE_DIR, "industry_map.json")
    ind_map = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                ind_map = json.load(f)
        except Exception:  # noqa: BLE001
            ind_map = {}

    if len(ind_map) < 1800:
        for p in sorted(glob.glob(os.path.join(OLD_REV_DIR, "*.csv"))):
            try:
                df = pd.read_csv(p, dtype=str)
                code_col = "stock_code" if "stock_code" in df.columns else ("stock_id" if "stock_id" in df.columns else None)
                if code_col and "industry" in df.columns:
                    for sid, ind in zip(df[code_col], df["industry"]):
                        if isinstance(ind, str) and ind.strip() and ind.strip() != "nan":
                            ind_map[str(sid).strip()] = ind.strip()
            except Exception:  # noqa: BLE001
                pass

    if len(ind_map) < 1950:
        today = dt.date.today()
        y, m = add_months(today.year, today.month, -2 if today.day < 10 else -1)
        for kind in ("sii", "otc"):
            for page in (0, 1):
                try:
                    r = _get(f"https://mopsov.twse.com.tw/nas/t21/{kind}/t21sc03_{y - 1911}_{m}_{page}.html")
                    dfs = pd.read_html(io.StringIO(r.content.decode("big5hkscs", errors="replace")))
                    cur_ind = None
                    for d in dfs:
                        if d.shape == (1, 2) and str(d.columns[0]).startswith("產業別"):
                            cur_ind = str(d.columns[0]).split("：")[-1].strip()
                        elif d.shape[1] >= 10 and cur_ind:
                            for _, row in d.iterrows():
                                code = str(row.iloc[0]).strip().removesuffix(".0")
                                if re.fullmatch(r"\d{4,6}", code):
                                    ind_map[code] = cur_ind
                except Exception:  # noqa: BLE001
                    pass
        if ind_map:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(ind_map, f, ensure_ascii=False)
    return ind_map


def load_shares(max_age_days=7):
    path = os.path.join(CACHE_DIR, "shares.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_days * 86400:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    out = {}
    try:
        for it in _get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L").json():
            out[str(it["公司代號"]).strip()] = _num(it.get("已發行普通股數或TDR原股發行股數"))
        for it in _get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O").json():
            out[str(it["SecuritiesCompanyCode"]).strip()] = _num(it.get("IssueShares"))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 取得發行股數失敗: {e}")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    out = {k: v for k, v in out.items() if v == v and v > 0}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out


# ---------------- 載入 external_lib 精確申報時間月營收 ----------------
def load_ext_revenue(start_ym, end_ym, industry_map):
    out = []
    y, m = start_ym
    while (y, m) <= tuple(end_ym):
        p = os.path.join(EXT_REV_DIR, f"revenue_{y}_{m:02d}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, dtype={"stock_code": str})
            if not df.empty:
                df = df.rename(columns={
                    "stock_code": "stock_id",
                    "company_name": "name",
                    "revenue_month": "ym",
                    "mom_pct": "MoM",
                    "yoy_pct": "YoY",
                })
                df["stock_id"] = df["stock_id"].astype(str).str.strip()
                df = df[df["stock_id"].str.fullmatch(r"\d{4}")].copy()
                if "industry" in df.columns:
                    df["industry"] = df["industry"].fillna("").astype(str).str.strip()
                    df["industry"] = np.where(df["industry"] != "", df["industry"], df["stock_id"].map(lambda s: industry_map.get(s, "")))
                else:
                    df["industry"] = df["stock_id"].map(lambda s: industry_map.get(s, ""))
                df["MoM"] = pd.to_numeric(df["MoM"], errors="coerce")
                df["YoY"] = pd.to_numeric(df["YoY"], errors="coerce")
                df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
                out.append(df[["ym", "stock_id", "name", "industry", "revenue", "MoM", "YoY", "announcement_time", "market_type"]])
        y, m = add_months(y, m, 1)
    if not out:
        return pd.DataFrame(columns=["ym", "stock_id", "name", "industry", "revenue", "MoM", "YoY", "announcement_time", "market_type"])
    return pd.concat(out, ignore_index=True)



# ---------------- 季報 EPS 下載與防未來函數處理 ----------------
def _eps_deadline(year: int, quarter: int) -> pd.Timestamp:
    if quarter == 1:
        return pd.Timestamp(year=year, month=5, day=15)
    if quarter == 2:
        return pd.Timestamp(year=year, month=8, day=14)
    if quarter == 3:
        return pd.Timestamp(year=year, month=11, day=14)
    return pd.Timestamp(year=year + 1, month=3, day=31)


def update_eps_cache(start_year: int, end_year: int):
    print(f"正在下載/更新 {start_year} ~ {end_year} 季報 EPS 資料（TWSE / TPEx OpenAPI）...")
    twse_urls = [
        "https://openapi.twse.com.tw/v1/opendata/t187ap06_L_ci",
        "https://openapi.twse.com.tw/v1/opendata/t187ap06_L_basi",
        "https://openapi.twse.com.tw/v1/opendata/t187ap06_L_bd",
        "https://openapi.twse.com.tw/v1/opendata/t187ap06_L_fh",
        "https://openapi.twse.com.tw/v1/opendata/t187ap06_L_ins",
        "https://openapi.twse.com.tw/v1/opendata/t187ap06_L_mim",
    ]
    tpex_urls = [
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_ci",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_basi",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_bd",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_fh",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_ins",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_mim",
    ]
    rows = []
    for url in twse_urls + tpex_urls:
        try:
            data = _get(url, tries=2, wait=3).json()
            if not isinstance(data, list):
                continue
            for it in data:
                code = str(it.get("公司代號") or it.get("SecuritiesCompanyCode") or "").strip()
                if not re.fullmatch(r"\d{4}", code):
                    continue
                yr_raw = str(it.get("年度") or it.get("Year") or "").strip()
                qt_raw = str(it.get("季別") or it.get("Quarter") or "").strip()
                eps_raw = (
                    it.get("基本每股盈餘（元）")
                    or it.get("基本每股盈餘(元)")
                    or it.get("基本每股盈餘")
                    or it.get("BasicEarningsPerShare")
                )
                if not (yr_raw.isdigit() and qt_raw.isdigit()):
                    continue
                yr = int(yr_raw)
                if yr < 1911:
                    yr += 1911
                qt = int(qt_raw)
                eps_val = _num(eps_raw)
                if start_year <= yr <= end_year and qt in (1, 2, 3, 4) and eps_val == eps_val:
                    rows.append((code, yr, qt, eps_val))
        except Exception:
            pass

    if not rows:
        print("OpenAPI 未回傳歷史季報（OpenAPI 通常僅提供最新一季）；若需完整多年歷史季報，可將 CSV 放入 data_cache/eps/。")
        return

    df_new = pd.DataFrame(rows, columns=["stock_id", "year", "quarter", "cum_eps"]).drop_duplicates(
        ["stock_id", "year", "quarter"], keep="last"
    )
    cache_file = os.path.join(EPS_DIR, "eps_quarterly.csv")
    if os.path.exists(cache_file):
        df_old = pd.read_csv(cache_file, dtype={"stock_id": str})
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            ["stock_id", "year", "quarter"], keep="last"
        )
    else:
        df_all = df_new
    df_all.sort_values(["stock_id", "year", "quarter"]).to_csv(cache_file, index=False, encoding="utf-8-sig")
    print(f"季報 EPS 快取已更新，目前共 {len(df_all):,} 筆季報紀錄（{cache_file}）")


def load_and_prepare_eps(dates: pd.DatetimeIndex) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(EPS_DIR, "*.csv")))
    if not files:
        return pd.DataFrame()
    frames = []
    for fp in files:
        try:
            d = pd.read_csv(fp, dtype={"stock_id": str})
            if not d.empty:
                frames.append(d)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")
    df = df.dropna(subset=["year", "quarter"]).copy()
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    df = df.drop_duplicates(["stock_id", "year", "quarter"], keep="last").sort_values(["stock_id", "year", "quarter"])

    if "eps" in df.columns:
        df["q_eps"] = pd.to_numeric(df["eps"], errors="coerce")
    elif "cum_eps" in df.columns:
        df["cum_eps"] = pd.to_numeric(df["cum_eps"], errors="coerce")
        g_yr = df.groupby(["stock_id", "year"], sort=False)
        prev_q = g_yr["quarter"].shift(1)
        prev_cum = g_yr["cum_eps"].shift(1)
        df["q_eps"] = np.where(
            df["quarter"] == 1,
            df["cum_eps"],
            np.where(prev_q == df["quarter"] - 1, df["cum_eps"] - prev_cum, np.nan),
        )
    else:
        return pd.DataFrame()

    df["_yq_num"] = df["year"] * 4 + df["quarter"]
    g = df.groupby("stock_id", sort=False)

    q0 = df["q_eps"]
    q1 = np.where(g["_yq_num"].shift(1) == df["_yq_num"] - 1, g["q_eps"].shift(1), np.nan)
    q2 = np.where(g["_yq_num"].shift(2) == df["_yq_num"] - 2, g["q_eps"].shift(2), np.nan)
    q3 = np.where(g["_yq_num"].shift(3) == df["_yq_num"] - 3, g["q_eps"].shift(3), np.nan)
    df["ttm_eps"] = q0 + q1 + q2 + q3

    yoy_prev = np.where(g["_yq_num"].shift(4) == df["_yq_num"] - 4, g["q_eps"].shift(4), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["eps_yoy"] = np.where(np.abs(yoy_prev) > 1e-9, (df["q_eps"] - yoy_prev) / np.abs(yoy_prev) * 100.0, np.nan)

    if "announcement_date" in df.columns:
        ann_dt = pd.to_datetime(df["announcement_date"], errors="coerce")
        fallback_dt = [_eps_deadline(y, q) for y, q in zip(df["year"], df["quarter"])]
        eff_dt = ann_dt.fillna(pd.Series(fallback_dt, index=df.index))
    else:
        eff_dt = pd.Series([_eps_deadline(y, q) for y, q in zip(df["year"], df["quarter"])], index=df.index)

    pos_next = dates.searchsorted(eff_dt, side="right")
    valid_pos = (pos_next >= 0) & (pos_next < len(dates))
    df = df[valid_pos].copy()
    df["avail_date"] = dates[pos_next[valid_pos]]
    return df[["avail_date", "stock_id", "q_eps", "ttm_eps", "eps_yoy"]].dropna(subset=["q_eps"])


# =====================================================================
# 二、市場矩陣 / 成長率差值（YoY_Diff & 連3月MoM_Diff）+ 當月YoY底線 / 資格判斷
# =====================================================================
class Market:
    def __init__(self, prices, vol_window):
        p = prices.drop_duplicates(["date", "stock_id"])
        piv = lambda c: p.pivot(index="date", columns="stock_id", values=c).sort_index()  # noqa: E731
        self.close, self.open, self.chg = piv("close"), piv("open"), piv("chg")
        self.high = piv("high") if "high" in p.columns else self.close.copy() * np.nan
        self.low = piv("low") if "low" in p.columns else self.close.copy() * np.nan
        self.volume = piv("volume").fillna(0.0)
        self.dates = self.close.index
        self.close_ff = self.close.ffill()
        prev = self.close_ff.shift(1)
        self.ref = (self.close - self.chg).fillna(prev)
        f = (self.ref / prev)
        etf_id = str(PARAMS.get("Idle_ETF_Code", "0050")).strip()
        valid_f = (f > 0.5) & (f < 1.5)
        if etf_id in f.columns:
            valid_f[etf_id] = (f[etf_id] > 0.2) & (f[etf_id] < 1.5)
        f = f.where(valid_f).fillna(1.0)
        self.cum = f.cumprod()                                   # 除權息還原因子
        self.avg_lots = (self.volume / 1000.0).rolling(vol_window, min_periods=max(5, vol_window // 2)).mean().shift(1)
        self.avg_turnover = (self.close * self.volume).rolling(vol_window, min_periods=max(5, vol_window // 2)).mean().shift(1)
        self.up_pct = (self.close / self.ref - 1) * 100          # 當日收盤漲跌幅 %
        self.gap_pct = (self.open / self.ref - 1) * 100          # 當日開盤漲跌幅 %
        self.low_pct = (self.low / self.ref - 1) * 100           # 當日最低漲跌幅 %


def prepare_revenue_table(rev_all: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    預先計算每筆月營收的：
      1. `YoY_Diff`   = 當月 YoY(t) - 上月 YoY(t-1)
      2. `MoM_Diff_0` = 當月 MoM(t) - 上月 MoM(t-1)
      3. `MoM_Diff_1` = 前1月 MoM(t-1) - 前2月 MoM(t-2)
      4. `MoM_Diff_2` = 前2月 MoM(t-2) - 前3月 MoM(t-3)
    （嚴格檢查月份連續無跳月，若有缺月則差值設為 NaN）
    並解析公告日期與時間字串。
    """
    df = rev_all.drop_duplicates(["ym", "stock_id"], keep="last").sort_values(["stock_id", "ym"]).copy()

    ym_parts = df["ym"].str.split("-", expand=True).astype(int)
    df["_ym_num"] = ym_parts[0] * 12 + ym_parts[1]
    g = df.groupby("stock_id", sort=False)

    ym_1 = g["_ym_num"].shift(1)
    ym_2 = g["_ym_num"].shift(2)
    ym_3 = g["_ym_num"].shift(3)

    yoy_1 = np.where(ym_1 == df["_ym_num"] - 1, g["YoY"].shift(1), np.nan)
    mom_1 = np.where(ym_1 == df["_ym_num"] - 1, g["MoM"].shift(1), np.nan)
    mom_2 = np.where(ym_2 == df["_ym_num"] - 2, g["MoM"].shift(2), np.nan)
    mom_3 = np.where(ym_3 == df["_ym_num"] - 3, g["MoM"].shift(3), np.nan)

    df["Prev_YoY"] = yoy_1
    df["Prev_MoM"] = mom_1
    df["YoY_Diff"] = df["YoY"] - df["Prev_YoY"]       # 當月 YoY - 上月 YoY
    df["MoM_Diff_0"] = df["MoM"] - mom_1              # 第 t 月：MoM(t) - MoM(t-1)
    df["MoM_Diff_1"] = mom_1 - mom_2                  # 第 t-1 月：MoM(t-1) - MoM(t-2)
    df["MoM_Diff_2"] = mom_2 - mom_3                  # 第 t-2 月：MoM(t-2) - MoM(t-3)

    df["not_excl"] = ~df["industry"].map(is_excluded_industry)

    s = df["announcement_time"].fillna("").astype(str).strip() if hasattr(df["announcement_time"].fillna("").astype(str), "strip") else df["announcement_time"].fillna("").astype(str).str.strip()
    valid_ann = (s != "") & (~s.str.endswith("00:00:00")) & (s.str.len() >= 19)
    df["valid_ann"] = valid_ann
    df["ann_date"] = pd.to_datetime(s.where(valid_ann, "").str.slice(0, 10), errors="coerce")
    df["ann_time_str"] = s.where(valid_ann, "").str.slice(11, 19)
    df["ann_day"] = df["ann_date"].dt.day.fillna(0).astype(int)

    pos_left = np.full(len(df), -1, dtype=int)
    pos_right = np.full(len(df), -1, dtype=int)
    is_trading_day = np.zeros(len(df), dtype=bool)

    mask = valid_ann & df["ann_date"].notna()
    if mask.any():
        ad = df.loc[mask, "ann_date"]
        pl = dates.searchsorted(ad, side="left")
        pr = dates.searchsorted(ad, side="right")
        ok_l = (pl >= 0) & (pl < len(dates))
        ok_r = (pr >= 0) & (pr < len(dates))
        pl_clean = np.where(ok_l, pl, -1)
        pr_clean = np.where(ok_r, pr, -1)
        pos_left[mask.to_numpy()] = pl_clean
        pos_right[mask.to_numpy()] = pr_clean

        valid_pl = pl_clean >= 0
        td_flags = np.zeros(len(ad), dtype=bool)
        if valid_pl.any():
            td_flags[valid_pl] = (dates[pl_clean[valid_pl]].to_numpy() == ad.to_numpy()[valid_pl])
        is_trading_day[mask.to_numpy()] = td_flags

    df["pos_left"] = pos_left
    df["pos_right"] = pos_right
    df["ann_is_trading_day"] = is_trading_day
    return df


def make_daily_signals(prep_rev: pd.DataFrame, dates: pd.DatetimeIndex, p: dict) -> dict:
    """
    v5 進場邏輯：
      * 底線條件（若 Min_YoY > -999）：`當月 YoY > Min_YoY`
      * 條件 X：`YoY_Diff = 當月 YoY - 上月 YoY > X_YoY_Diff`
      * 條件 Y：`MoM_Diff_0 > Y_MoM_Diff` 且 `MoM_Diff_1 > Y_MoM_Diff` 且 `MoM_Diff_2 > Y_MoM_Diff`
      * 排序：同一交易日內，依執行時序（open -> intraday -> close）優先，同時段內依 `YoY_Diff` 由高到低排序。
    """
    min_ann_day = int(p.get("Min_Ann_Day", 1))
    valid = prep_rev[prep_rev["valid_ann"] & prep_rev["not_excl"]]
    if min_ann_day > 1:
        valid = valid[valid["ann_day"] >= min_ann_day]

    # v5 新增：當月 YoY 必須大於 Min_YoY 底線
    min_yoy = p.get("Min_YoY")
    if min_yoy is not None and float(min_yoy) > -999.0:
        valid = valid[valid["YoY"] > float(min_yoy)]
    max_yoy = p.get("Max_YoY")
    if max_yoy is not None and float(max_yoy) < 99999.0:
        valid = valid[valid["YoY"] <= float(max_yoy)]

    if valid.empty:
        return {}

    x_diff = float(p["X_YoY_Diff"])
    y_diff = float(p["Y_MoM_Diff"])
    cx = valid["YoY_Diff"] > x_diff
    cy = (valid["MoM_Diff_0"] > y_diff) & (valid["MoM_Diff_1"] > y_diff) & (valid["MoM_Diff_2"] > y_diff)
    hit = valid[cx | cy].copy()
    if hit.empty:
        return {}

    cx_h, cy_h = cx.loc[hit.index], cy.loc[hit.index]
    hit["reason"] = np.where(cx_h & cy_h, "YoY_Diff+MoM_Diff3", np.where(cx_h, "YoY_Diff", "MoM_Diff3"))

    pre_end = str(p.get("Pre_Market_End", "09:00:00"))
    cutoff = str(p.get("Cutoff_Time", "13:25:00"))
    pre_mode = str(p.get("Pre_Market_Entry", "open")).lower()
    intra_mode = str(p.get("Intraday_Entry", "immediate")).lower()
    post_mode = str(p.get("Post_Market_Entry", "next_open")).lower()

    t_str = hit["ann_time_str"]
    is_td = hit["ann_is_trading_day"]

    cond_pre = (~is_td) | (t_str < pre_end)
    cond_intra = is_td & (t_str >= pre_end) & (t_str <= cutoff)
    cond_post = is_td & (t_str > cutoff)

    hit["ann_session"] = np.where(cond_pre, "pre_market", np.where(cond_intra, "intraday", "post_market"))

    sig_idx = np.full(len(hit), -1, dtype=int)
    exec_type = np.full(len(hit), "", dtype=object)

    pl = hit["pos_left"].to_numpy()
    pr = hit["pos_right"].to_numpy()

    if pre_mode == "open":
        sig_idx = np.where(cond_pre, pl, sig_idx)
        exec_type = np.where(cond_pre, "open", exec_type)
    elif pre_mode == "close":
        sig_idx = np.where(cond_pre, pl, sig_idx)
        exec_type = np.where(cond_pre, "close", exec_type)

    if intra_mode in ("immediate", "mid", "hl2", "open"):
        sig_idx = np.where(cond_intra, pl, sig_idx)
        exec_type = np.where(cond_intra, "intraday", exec_type)
    elif intra_mode == "close":
        sig_idx = np.where(cond_intra, pl, sig_idx)
        exec_type = np.where(cond_intra, "close", exec_type)
    elif intra_mode == "next_open":
        sig_idx = np.where(cond_intra, pr, sig_idx)
        exec_type = np.where(cond_intra, "open", exec_type)

    if post_mode == "next_open":
        sig_idx = np.where(cond_post, pr, sig_idx)
        exec_type = np.where(cond_post, "open", exec_type)
    elif post_mode == "next_close":
        sig_idx = np.where(cond_post, pr, sig_idx)
        exec_type = np.where(cond_post, "close", exec_type)

    hit["sig_idx"] = sig_idx
    hit["exec_type"] = exec_type
    hit = hit[(hit["sig_idx"] >= 0) & (hit["sig_idx"] < len(dates)) & (hit["exec_type"] != "")].copy()
    if hit.empty:
        return {}

    order_map = {"open": 0, "intraday": 1, "close": 2}
    hit["_exec_order"] = hit["exec_type"].map(order_map).fillna(9)
    hit = hit.sort_values(["sig_idx", "_exec_order", "YoY_Diff"], ascending=[True, True, False], na_position="last")

    cols = [
        "stock_id", "name", "industry", "ym",
        "YoY", "Prev_YoY", "YoY_Diff",
        "MoM", "Prev_MoM", "MoM_Diff_0", "MoM_Diff_1", "MoM_Diff_2",
        "announcement_time", "ann_day", "ann_session", "exec_type", "reason",
    ]
    sigs = {}
    for sig_i, grp in hit.groupby("sig_idx", sort=True):
        sigs[int(sig_i)] = grp[cols].drop_duplicates("stock_id", keep="first").reset_index(drop=True)
    return sigs


class Eligibility:
    def __init__(self, mk: Market, shares_map: dict, industry_map: dict, eps_df: pd.DataFrame = None):
        self.mk = mk
        self.cols = {s: j for j, s in enumerate(mk.close.columns)}
        self.shares = np.array([shares_map.get(s, np.nan) for s in mk.close.columns], dtype=float)
        self.ok_univ = np.array([not is_excluded_industry(industry_map.get(s, "")) for s in mk.close.columns])
        self._rank = {}
        self.has_eps = False
        self.last_q_eps = None
        self.ttm_eps = None
        self.eps_yoy = None
        if eps_df is not None and not eps_df.empty:
            dedup = eps_df.sort_values(["stock_id", "avail_date"]).drop_duplicates(["avail_date", "stock_id"], keep="last")
            for col_name, attr_name in [("q_eps", "last_q_eps"), ("ttm_eps", "ttm_eps"), ("eps_yoy", "eps_yoy")]:
                piv = dedup.pivot(index="avail_date", columns="stock_id", values=col_name)
                aligned = piv.reindex(index=self.mk.dates, columns=self.mk.close.columns).ffill()
                setattr(self, attr_name, aligned.to_numpy(dtype=float))
            self.has_eps = True

    def cap_pct(self, i: int, mode: str = "adj_cap") -> np.ndarray:
        if i < 0:
            i = 0
        key = (mode, i)
        if key not in self._rank:
            if mode == "turnover":
                metric = self.mk.avg_turnover.iloc[i].to_numpy(dtype=float)
                valid = self.ok_univ & np.isfinite(metric) & (metric > 0)
            elif mode == "adj_cap":
                close = self.mk.close.iloc[i].to_numpy(dtype=float)
                cum_latest = self.mk.cum.iloc[-1].to_numpy(dtype=float)
                cum_i = self.mk.cum.iloc[i].to_numpy(dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    est_shares = self.shares * (cum_latest / cum_i)
                metric = close * est_shares
                valid = self.ok_univ & np.isfinite(metric) & (close > 0) & (metric > 0)
            else:
                close = self.mk.close.iloc[i].to_numpy(dtype=float)
                metric = close * self.shares
                valid = self.ok_univ & np.isfinite(metric) & (close > 0)
            rank = np.full(len(metric), np.nan)
            if valid.any():
                rank[valid] = pd.Series(metric[valid]).rank(method="average").to_numpy() / valid.sum() * 100
            self._rank[key] = rank
        return self._rank[key]

    def ok(self, i: int, sid: str, p: dict, exec_type: str = "close", budget: float = 0.0) -> bool:
        j = self.cols.get(sid)
        if j is None:
            return False
        min_vol = float(p.get("Min_Avg_Volume", 0))
        if min_vol > 0 and not (self.mk.avg_lots.iat[i, j] >= min_vol):
            return False
        min_to = float(p.get("Min_Avg_Turnover", 0))
        max_to_pct = float(p.get("Max_Turnover_Pct", 0.0))
        if max_to_pct > 0 and budget > 0:
            min_to = max(min_to, budget / max_to_pct)
        if min_to > 0 and not (self.mk.avg_turnover.iat[i, j] >= min_to):
            return False

        if p.get("Enable_EPS_Filter", False) and self.has_eps:
            q_eps = self.last_q_eps[i, j]
            ttm = self.ttm_eps[i, j]
            e_yoy = self.eps_yoy[i, j]

            min_q = float(p.get("Min_Last_Q_EPS", -999.0))
            if min_q > -999.0 and not (q_eps == q_eps and q_eps > min_q):
                return False

            min_ttm = float(p.get("Min_TTM_EPS", -999.0))
            if min_ttm > -999.0 and not (ttm == ttm and ttm > min_ttm):
                return False

            min_eyoy = float(p.get("Min_Q_EPS_YoY", -999.0))
            if min_eyoy > -999.0 and not (e_yoy == e_yoy and e_yoy > min_eyoy):
                return False

            max_pe = float(p.get("Max_PE_Ratio", 999.0))
            if max_pe < 999.0:
                ref_px = self.mk.open.iat[i, j] if exec_type == "open" else self.mk.close.iat[i, j]
                if not (ttm == ttm and ttm > 0 and ref_px > 0 and (ref_px / ttm) <= max_pe):
                    return False

        lo, hi = (p["Cap_Decile_Low"] - 1) * 10, p["Cap_Decile_High"] * 10
        sz_mode = str(p.get("Size_Mode", "adj_cap")).lower()
        if exec_type == "open":
            if not (self.mk.open.iat[i, j] > 0):
                return False
            idx_rank = i if sz_mode == "turnover" else max(0, i - 1)
            r = self.cap_pct(idx_rank, sz_mode)[j]
            if not (r == r and lo < r <= hi):
                return False
            o_px, h_px, l_px, c_px = self.mk.open.iat[i, j], self.mk.high.iat[i, j], self.mk.low.iat[i, j], self.mk.close.iat[i, j]
            is_locked_open = (o_px == h_px == l_px == c_px) and (self.mk.gap_pct.iat[i, j] >= 9.0)
            return not (self.mk.gap_pct.iat[i, j] >= p["Limit_Pct"] or is_locked_open)
        else:
            if not (self.mk.close.iat[i, j] > 0):
                return False
            r = self.cap_pct(i, sz_mode)[j]
            if not (r == r and lo < r <= hi):
                return False
            h_px, c_px = self.mk.high.iat[i, j], self.mk.close.iat[i, j]
            is_locked_close = (c_px == h_px) and (self.mk.up_pct.iat[i, j] >= 9.0)
            return not (self.mk.up_pct.iat[i, j] >= p["Limit_Pct"] or is_locked_close)

    def buy_day_chg_ok(self, i: int, sid: str, p: dict, exec_type: str = "close") -> bool:
        min_chg = p.get("Min_Buy_Day_Chg")
        if min_chg is None or float(min_chg) <= -50.0:
            return True
        j = self.cols[sid]
        u = self.mk.gap_pct.iat[i, j] if exec_type == "open" else self.mk.up_pct.iat[i, j]
        return bool(u == u and u >= float(min_chg))

    def anti_chase_ok(self, i: int, sid: str, p: dict, exec_type: str = "close") -> bool:
        j = self.cols[sid]
        max_chg = p.get("Max_Buy_Day_Chg")
        if max_chg is not None and float(max_chg) < 50.0:
            u = self.mk.gap_pct.iat[i, j] if exec_type == "open" else self.mk.up_pct.iat[i, j]
            if not (u == u and u <= float(max_chg)):
                return False

        max_runup = p.get("Max_Pre_Runup_Pct")
        lookback = int(p.get("Pre_Runup_Days", 3))
        if max_runup is not None and float(max_runup) < 999.0 and lookback >= 1 and i >= lookback:
            base_c = self.mk.close_ff.iat[i - lookback, j]
            base_cum = self.mk.cum.iat[i - lookback, j]
            cur_cum = self.mk.cum.iat[i, j]
            if base_c > 0 and base_cum > 0 and cur_cum > 0:
                base_adj = base_c * (cur_cum / base_cum)
                cur_px = _entry_raw_price(self.mk, i, j, exec_type, str(p.get("Intraday_Price_Proxy", "mid")).lower())
                if cur_px > 0:
                    runup_pct = (cur_px / base_adj - 1.0) * 100.0
                    if runup_pct > float(max_runup):
                        return False
        return True


# =====================================================================
# 三、回測引擎（支援同日開盤賣出 -> 開盤買進 -> 盤中停損 -> 盤中買進 -> 收盤買進）
# =====================================================================
def target_budget(cur_equity: float, p: dict) -> float:
    pct = float(p.get("Position_Pct", 0.0))
    return cur_equity * (pct / 100.0) if pct > 0 else float(p["Position_Size"])


def lots_for_budget(price: float, budget: float) -> int:
    return int(budget // (price * 1000)) if (price and price > 0 and budget > 0) else 0


def _entry_raw_price(mk: Market, i: int, j: int, exec_type: str, intra_proxy: str) -> float:
    o = mk.open.iat[i, j]
    c = mk.close.iat[i, j]
    if exec_type == "open":
        return float(o)
    if exec_type == "intraday":
        if intra_proxy == "open":
            return float(o)
        if intra_proxy == "close":
            return float(c)
        if intra_proxy == "hl2":
            h, l = mk.high.iat[i, j], mk.low.iat[i, j]
            if h == h and l == l and h > 0 and l > 0:
                return float((h + l) * 0.5)
        if o > 0 and c > 0:
            return float((o + c) * 0.5)
        return float(c if c > 0 else o)
    return float(c)


def _record_sell(trades: list, sid: str, z: dict, d: pd.Timestamp, sell_px: float, held: int,
                 exit_reason: str, fee_rate: float, tax_rate: float) -> float:
    gross = sell_px * z["shares"]
    net = gross - max(20.0, gross * fee_rate) - gross * tax_rate
    trades.append({
        "side": "LONG", "stock_id": sid, "name": z["name"], "reason": z["reason"], "ym": z["ym"],
        "YoY": round(z["YoY"], 2) if z["YoY"] == z["YoY"] else np.nan,
        "YoY_Diff": round(z["YoY_Diff"], 2) if z["YoY_Diff"] == z["YoY_Diff"] else np.nan,
        "MoM": round(z["MoM"], 2) if z["MoM"] == z["MoM"] else np.nan,
        "MoM_Diff": round(z["MoM_Diff"], 2) if z["MoM_Diff"] == z["MoM_Diff"] else np.nan,
        "announcement_time": z["ann_time"], "ann_session": z["ann_session"],
        "entry_exec": z["exec_type"], "buy_date": z["buy_date"],
        "buy_day_chg_pct": round(z["buy_chg"], 2) if z["buy_chg"] == z["buy_chg"] else np.nan,
        "buy_price": round(z["px"], 2), "sell_date": d.date(), "sell_price": round(sell_px, 2),
        "lots": z["shares"] // 1000, "hold_days": held, "exit": exit_reason,
        "peak_ret_pct": round((z["peak"] / z["px"] - 1) * 100, 2),
        "pnl": round(net - z["cost"], 0), "return_pct": round((net / z["cost"] - 1) * 100, 3),
    })
    return net


def _record_short_cover(trades: list, sid: str, sz: dict, d: pd.Timestamp, cover_px: float, held: int,
                        exit_reason: str, fee_rate: float, tax_rate: float) -> float:
    sh = sz["shares"]
    short_gross = sz["short_px"] * sh
    short_net = short_gross - max(20.0, short_gross * fee_rate) - short_gross * tax_rate
    cover_gross = cover_px * sh
    cover_cost = cover_gross + max(20.0, cover_gross * fee_rate)
    pnl = short_net - cover_cost
    trades.append({
        "side": "SHORT", "stock_id": sid, "name": sz["name"], "reason": sz["reason"], "ym": sz["ym"],
        "YoY": round(sz["YoY"], 2) if sz["YoY"] == sz["YoY"] else np.nan,
        "YoY_Diff": round(sz["YoY_Diff"], 2) if sz["YoY_Diff"] == sz["YoY_Diff"] else np.nan,
        "MoM": round(sz["MoM"], 2) if sz["MoM"] == sz["MoM"] else np.nan,
        "MoM_Diff": round(sz["MoM_Diff"], 2) if sz["MoM_Diff"] == sz["MoM_Diff"] else np.nan,
        "announcement_time": sz["ann_time"], "ann_session": sz["ann_session"],
        "entry_exec": "gap_short_open", "buy_date": sz["short_date"],
        "buy_day_chg_pct": round(sz["gap_chg"], 2) if sz["gap_chg"] == sz["gap_chg"] else np.nan,
        "buy_price": round(sz["short_px"], 2), "sell_date": d.date(), "sell_price": round(cover_px, 2),
        "lots": sh // 1000, "hold_days": held, "exit": exit_reason, "peak_ret_pct": np.nan,
        "pnl": round(pnl, 0), "return_pct": round((pnl / short_gross) * 100.0, 3),
    })
    return pnl


def simulate(mk: Market, sigs: dict, elig: Eligibility, p: dict, bt: dict, start: pd.Timestamp):
    fee_rate = 0.001425 * bt["Fee_Discount"]
    slip, tax_std, tax_dt = bt["Slippage"], bt["Tax"], bt.get("DayTrade_Tax", 0.0015)
    cash = float(bt["Initial_Capital"])
    pos, short_pos, trades, equity = {}, {}, [], []
    stats = {"eligible": 0, "skip_buy_chg": 0, "skip_chase": 0, "buy_open": 0, "buy_intraday": 0, "buy_close": 0,
             "gap_short_cnt": 0, "gap_short_sl": 0, "gap_short_limit_next": 0, "gap_short_reverse_long": 0,
             "etf_buy_cnt": 0, "etf_sell_cnt": 0, "etf_sl_cnt": 0, "etf_pnl": 0.0}
    en_etf = bool(p.get("Enable_Idle_ETF", False))
    etf_code = str(p.get("Idle_ETF_Code", "0050")).strip()
    etf_j = elig.cols.get(etf_code)
    etf_sma_n = int(p.get("ETF_SMA_Days", 60))
    etf_sl = float(p.get("ETF_Stop_Loss_Pct", 7.0)) / 100.0
    etf_buf = float(p.get("ETF_Cash_Buffer_Pct", 5.0)) / 100.0
    etf_pos = {"shares": 0, "avg_px": 0.0, "cost": 0.0, "cum": 1.0, "buy_date": None, "hold_i": 0}
    etf_stopped_wait_reclaim = False
    etf_adj_close = (mk.close_ff[etf_code] / mk.cum[etf_code]) if (en_etf and etf_j is not None) else None
    etf_sma_series = etf_adj_close.rolling(etf_sma_n, min_periods=etf_sma_n).mean() if (etf_adj_close is not None and etf_sma_n > 0) else None
    N, sl = int(p["N_Days"]), float(p["Stop_Loss_Pct"]) / 100.0
    sl_mode = str(p.get("Stop_Loss_Mode", "intraday")).lower()
    raw_cap = p.get("Dyn_SL_Cap_Pct")
    use_dyn_sl = (raw_cap is not None and float(raw_cap) > -999.0)
    dyn_cap = float(raw_cap) / 100.0 if use_dyn_sl else -9.99
    tr, act = float(p.get("Trail_Stop_Pct", 0.0)) / 100.0, float(p.get("Trail_Activate_Pct", 0.0)) / 100.0
    intra_proxy, max_pos = str(p.get("Intraday_Price_Proxy", "mid")).lower(), int(p["Max_Positions"])
    en_gs = bool(p.get("Enable_Gap_Short", True))
    gs_min, gs_max = float(p.get("Gap_Short_Min_Pct", 3.0)), float(p.get("Gap_Short_Max_Pct", 8.5))
    short_sl = float(p.get("Short_Stop_Loss_Pct", 4.0)) / 100.0
    fired_ym_sid = set()

    def _cur_eq(di: int) -> float:
        lv = 0.0
        for s, z in pos.items():
            jj = elig.cols[s]
            px_n = mk.close_ff.iat[di, jj] / (mk.cum.iat[di, jj] / z["cum"])
            lv += z["shares"] * px_n if (px_n == px_n and px_n > 0) else z["cost"]
        sp = 0.0
        for s, sz in short_pos.items():
            jj = elig.cols[s]
            px_n = mk.close_ff.iat[di, jj] / (mk.cum.iat[di, jj] / sz["cum"])
            if px_n == px_n and px_n > 0:
                sp += (sz["short_px"] - px_n) * sz["shares"]
        ev = 0.0
        if etf_pos["shares"] > 0 and etf_j is not None:
            epx = mk.close_ff.iat[di, etf_j] / (mk.cum.iat[di, etf_j] / etf_pos["cum"])
            ev = etf_pos["shares"] * epx if (epx == epx and epx > 0) else etf_pos["cost"]
        return cash + lv + sp + ev

    def _avail_cash(extra_shorts: dict = None) -> float:
        locked_short = sum(sz["short_px"] * sz["shares"] for sz in short_pos.values())
        if extra_shorts:
            locked_short += sum(sz["short_px"] * sz["shares"] for sz in extra_shorts.values())
        return cash - locked_short

    def _buy_one(i: int, d: pd.Timestamp, r, ex_type: str, rev_short: bool = False) -> bool:
        nonlocal cash, etf_pos
        sid, ym_key = r.stock_id, (r.ym, r.stock_id)
        if (len(pos) + len(short_pos)) >= max_pos or sid in pos or sid in short_pos or ym_key in fired_ym_sid:
            return False
        cur_budget = target_budget(_cur_eq(i), p)
        if not elig.ok(i, sid, p, exec_type=ex_type, budget=cur_budget):
            return False
        if not rev_short:
            stats["eligible"] += 1
        if (not rev_short) or bool(p.get("Reverse_Check_Buy_Chg", False)):
            if not elig.buy_day_chg_ok(i, sid, p, exec_type=ex_type):
                if not rev_short:
                    stats["skip_buy_chg"] += 1
                return False
            if not elig.anti_chase_ok(i, sid, p, exec_type=ex_type):
                if not rev_short:
                    stats["skip_chase"] += 1
                return False
        j = elig.cols[sid]
        raw_px = _entry_raw_price(mk, i, j, ex_type, intra_proxy)
        if not (raw_px > 0):
            return False
        px = raw_px * (1 + slip)
        lots = lots_for_budget(px, cur_budget)
        if lots < 1:
            return False
        sh = lots * 1000
        cost = px * sh + max(20.0, px * sh * fee_rate)
        if _avail_cash() < cost and en_etf and etf_j is not None and etf_pos["shares"] > 0:
            need_cash = cost - _avail_cash()
            raw_epx = float(mk.open.iat[i, etf_j] if ex_type == "open" else mk.close.iat[i, etf_j])
            if raw_epx > 0:
                adj_e = mk.cum.iat[i, etf_j] / etf_pos["cum"]
                sell_epx = (raw_epx / adj_e) * (1 - slip)
                cur_market_lots = int((etf_pos["shares"] / adj_e) // 1000)
                raw_sell_px = raw_epx * (1 - slip)
                sell_lots = min(cur_market_lots, int(np.ceil(need_cash / (raw_sell_px * 1000 * (1 - fee_rate - 0.001)))))
                if sell_lots >= 1:
                    sell_sh = (sell_lots * 1000) * adj_e
                    gross_e = sell_epx * sell_sh
                    net_e = gross_e - max(20.0, gross_e * fee_rate) - gross_e * 0.001
                    part_cost = etf_pos["cost"] * (sell_sh / etf_pos["shares"])
                    pnl_e = net_e - part_cost
                    cash += net_e
                    stats["etf_sell_cnt"] += 1
                    stats["etf_pnl"] += pnl_e
                    trades.append({
                        "side": "ETF", "stock_id": etf_code, "name": "元大台灣50", "reason": "Idle_ETF", "ym": str(d.date())[:7],
                        "YoY": np.nan, "YoY_Diff": np.nan, "MoM": np.nan, "MoM_Diff": np.nan,
                        "announcement_time": "", "ann_session": "idle", "entry_exec": ex_type,
                        "buy_date": etf_pos["buy_date"], "buy_day_chg_pct": np.nan,
                        "buy_price": round(etf_pos["avg_px"], 2), "sell_date": d.date(), "sell_price": round(sell_epx, 2),
                        "lots": sell_lots, "hold_days": i - etf_pos["hold_i"], "exit": "etf_for_stock",
                        "peak_ret_pct": np.nan, "pnl": round(pnl_e, 0), "return_pct": round((net_e / part_cost - 1) * 100, 3)
                    })
                    etf_pos["shares"] -= sell_sh
                    etf_pos["cost"] -= part_cost
                    if etf_pos["shares"] <= 0:
                        etf_pos = {"shares": 0, "avg_px": 0.0, "cost": 0.0, "cum": 1.0, "buy_date": None, "hold_i": 0}
        if _avail_cash() < cost:
            return False
        cash -= cost
        fired_ym_sid.add(ym_key)
        stats["buy_" + ex_type] += 1
        buy_chg = float(mk.gap_pct.iat[i, j] if ex_type == "open" else mk.up_pct.iat[i, j])
        pos[sid] = {"i": i, "buy_date": d.date(), "shares": sh, "px": px, "cost": cost, "cum": mk.cum.iat[i, j],
                    "peak": px, "cur_sl_pct": -sl, "exit": "", "name": r.name, "reason": r.reason, "ym": r.ym, "YoY": r.YoY,
                    "YoY_Diff": r.YoY_Diff, "MoM": r.MoM, "MoM_Diff": r.MoM_Diff_0,
                    "ann_time": r.announcement_time, "ann_session": r.ann_session, "exec_type": ex_type, "buy_chg": buy_chg}
        return True

    for i, d in enumerate(mk.dates):
        if d < start:
            continue
        day_sigs = sigs.get(i)
        intraday_shorts, rev_longs, handled_today = {}, [], set()

        # 1) 開盤賣出到期/停損多單
        for sid in list(pos):
            z = pos[sid]
            held = i - z["i"]
            er = z["exit"] or ("time" if held >= N else "")
            if er:
                j = elig.cols[sid]
                o = mk.open.iat[i, j]
                if o > 0 and mk.gap_pct.iat[i, j] > -p["Limit_Pct"]:
                    cash += _record_sell(trades, sid, z, d, (o / (mk.cum.iat[i, j] / z["cum"])) * (1 - slip), held, er, fee_rate, tax_std)
                    del pos[sid]

        # 1a) 0050 獨立風控檢查（開盤前判斷昨日是否跌破 SMA，若破線則開盤出清轉現金）
        if en_etf and etf_j is not None and i > 0:
            y_c = float(etf_adj_close.iat[i - 1])
            y_sma = float(etf_sma_series.iat[i - 1]) if etf_sma_series is not None else np.nan
            y_sma_prev = float(etf_sma_series.iat[i - 2]) if (etf_sma_series is not None and i >= 2) else y_sma
            etf_trend_ok = True if etf_sma_n <= 0 else (y_c == y_c and y_sma == y_sma and y_c > y_sma and y_sma >= y_sma_prev)
            if not etf_trend_ok:
                etf_stopped_wait_reclaim = False  # 均線已破，重置停損冷卻狀態
                if etf_pos["shares"] > 0 and mk.open.iat[i, etf_j] > 0:
                    adj_e = mk.cum.iat[i, etf_j] / etf_pos["cum"]
                    s_px = (float(mk.open.iat[i, etf_j]) / adj_e) * (1 - slip)
                    gross_e = s_px * etf_pos["shares"]
                    net_e = gross_e - max(20.0, gross_e * fee_rate) - gross_e * 0.001
                    pnl_e = net_e - etf_pos["cost"]
                    cash += net_e
                    stats["etf_sell_cnt"] += 1
                    stats["etf_pnl"] += pnl_e
                    trades.append({
                        "side": "ETF", "stock_id": etf_code, "name": "元大台灣50", "reason": "Idle_ETF", "ym": str(d.date())[:7],
                        "YoY": np.nan, "YoY_Diff": np.nan, "MoM": np.nan, "MoM_Diff": np.nan,
                        "announcement_time": "", "ann_session": "idle", "entry_exec": "open",
                        "buy_date": etf_pos["buy_date"], "buy_day_chg_pct": np.nan,
                        "buy_price": round(etf_pos["avg_px"], 2), "sell_date": d.date(), "sell_price": round(s_px, 2),
                        "lots": etf_pos["shares"] // 1000, "hold_days": i - etf_pos["hold_i"], "exit": "etf_sma_exit",
                        "peak_ret_pct": np.nan, "pnl": round(pnl_e, 0), "return_pct": round((net_e / etf_pos["cost"] - 1) * 100, 3)
                    })
                    etf_pos = {"shares": 0, "avg_px": 0.0, "cost": 0.0, "cum": 1.0, "buy_date": None, "hold_i": 0}

        # 1b) 開盤強制回補「前一日因鎖漲停留倉」的空單（一般稅率 0.3%）
        for sid in list(short_pos):
            sz = short_pos[sid]
            j = elig.cols[sid]
            o = mk.open.iat[i, j]
            if o > 0 and mk.gap_pct.iat[i, j] < p["Limit_Pct"]:
                cover_px = (o / (mk.cum.iat[i, j] / sz["cum"])) * (1 + slip)
                cash += _record_short_cover(trades, sid, sz, d, cover_px, i - sz["i"], "short_limit_next_open", fee_rate, tax_std)
                del short_pos[sid]

        # 2) 開盤檢查：針對「盤前/前日盤後公告」檢查是否跳空 3%~8.5% 放空；其餘才走正常開盤買多
        if day_sigs is not None:
            if en_gs:
                pre_post_sigs = day_sigs[day_sigs["ann_session"].isin(["pre_market", "post_market"])]
                for r in pre_post_sigs.itertuples():
                    sid = r.stock_id
                    if (len(pos) + len(short_pos) + len(intraday_shorts)) >= max_pos:
                        break
                    cur_budget = target_budget(_cur_eq(i), p)
                    if sid in pos or sid in short_pos or (r.ym, sid) in fired_ym_sid or not elig.ok(i, sid, p, "open", budget=cur_budget):
                        continue
                    j = elig.cols[sid]
                    gap = float(mk.gap_pct.iat[i, j])
                    if (gs_min < gap < gs_max) and mk.open.iat[i, j] > 0:
                        s_px = float(mk.open.iat[i, j]) * (1 - slip)
                        lots = lots_for_budget(s_px, cur_budget)
                        if lots >= 1 and _avail_cash(intraday_shorts) >= s_px * lots * 1000 * 1.15:
                            stats["gap_short_cnt"] += 1
                            handled_today.add(sid)
                            intraday_shorts[sid] = {"i": i, "short_date": d.date(), "shares": lots * 1000, "short_px": s_px,
                                                    "cum": mk.cum.iat[i, j], "name": r.name, "reason": r.reason, "ym": r.ym,
                                                    "YoY": r.YoY, "YoY_Diff": r.YoY_Diff, "MoM": r.MoM, "MoM_Diff": r.MoM_Diff_0,
                                                    "ann_time": r.announcement_time, "ann_session": r.ann_session, "gap_chg": gap, "row_obj": r}

            for r in day_sigs[day_sigs["exec_type"] == "open"].itertuples():
                if r.stock_id not in handled_today:
                    _buy_one(i, d, r, "open", False)

        # 3) 多單盤中觸價停損（開盤跳空創高先更新停損線，再檢查開盤/盤中觸價，並清除冗餘分支）
        if sl > 0 and sl_mode == "intraday":
            for sid in list(pos):
                z = pos[sid]
                held = i - z["i"]
                if held <= 0:
                    continue
                j = elig.cols[sid]
                o = mk.open.iat[i, j]
                if o > 0 and mk.gap_pct.iat[i, j] > -p["Limit_Pct"]:
                    adj = mk.cum.iat[i, j] / z["cum"]
                    ov = o / adj
                    if use_dyn_sl and ov > z["peak"]:
                        z["peak"] = ov
                        open_gain = ov / z["px"] - 1.0
                        if open_gain > 0:
                            target_sl = min(-sl + open_gain, dyn_cap)
                            if target_sl > z.get("cur_sl_pct", -sl):
                                z["cur_sl_pct"] = target_sl
                    cur_sl = z.get("cur_sl_pct", -sl)
                    sv = z["px"] * (1.0 + cur_sl)
                    is_be = (cur_sl > -sl + 1e-9)
                    tag_gap = "be_gap" if is_be else "stop_gap"
                    tag_intra = "be_intraday" if is_be else "stop_intraday"
                    if ov <= sv:
                        cash += _record_sell(trades, sid, z, d, ov * (1 - slip), held, tag_gap, fee_rate, tax_std)
                        del pos[sid]
                    elif mk.low.iat[i, j] > 0 and (mk.low.iat[i, j] / adj) <= sv:
                        cash += _record_sell(trades, sid, z, d, sv * (1 - slip), held, tag_intra, fee_rate, tax_std)
                        del pos[sid]

        # 3b) 功能 2 當日空單結算（盤中停損與漲停皆不反手；僅正常收盤回補才反手作多）
        for sid, sz in intraday_shorts.items():
            j = elig.cols[sid]
            hi, cl, up_d = float(mk.high.iat[i, j]), float(mk.close.iat[i, j]), float(mk.up_pct.iat[i, j])
            s_stop = sz["short_px"] * (1.0 + short_sl) if short_sl > 0 else np.inf
            lim_up = (up_d >= p["Limit_Pct"])

            if short_sl > 0 and hi == hi and hi >= s_stop and sz["gap_chg"] < p["Limit_Pct"]:
                cash += _record_short_cover(trades, sid, sz, d, s_stop * (1 + slip), 0, "short_stop_intraday", fee_rate, tax_dt)
                fired_ym_sid.add((sz["ym"], sid))
                stats["gap_short_sl"] += 1
            elif lim_up:
                short_pos[sid] = sz
                fired_ym_sid.add((sz["ym"], sid))
                stats["gap_short_limit_next"] += 1
            else:
                cash += _record_short_cover(trades, sid, sz, d, cl * (1 + slip), 0, "short_eod_cover", fee_rate, tax_dt)
                rev_longs.append(sz["row_obj"])

        # 4) 盤中買進、收盤買進（排除今日已做跳空放空處理者），以及正常回補後的反手作多
        if day_sigs is not None:
            for r in day_sigs[day_sigs["exec_type"].isin(["intraday", "close"])].itertuples():
                if r.stock_id not in handled_today:
                    _buy_one(i, d, r, r.exec_type, False)
        for r in rev_longs:
            if _buy_one(i, d, r, "close", True):
                stats["gap_short_reverse_long"] += 1

        # 5) 盤後估值與多單收盤停損/移動停利判斷
        hv = 0.0
        for sid, z in pos.items():
            j = elig.cols[sid]
            adj = mk.cum.iat[i, j] / z["cum"]
            v = mk.close_ff.iat[i, j] / adj
            hv += z["shares"] * v
            held = i - z["i"]
            if (held > 0) or (z["exec_type"] == "open"):
                hr = mk.high.iat[i, j]
                z["peak"] = max(z["peak"], v, (hr / adj) if (hr == hr and hr > 0) else v)
                if use_dyn_sl and sl > 0:
                    peak_gain = z["peak"] / z["px"] - 1.0
                    if peak_gain > 0:
                        target_sl = min(-sl + peak_gain, dyn_cap)
                        if target_sl > z.get("cur_sl_pct", -sl):
                            z["cur_sl_pct"] = target_sl
                if not z["exit"]:
                    lr = mk.low.iat[i, j]
                    need_c = (sl_mode == "close") or not (lr == lr and lr > 0) or (held == 0 and z["exec_type"] == "open")
                    cur_sl = z.get("cur_sl_pct", -sl)
                    if sl > 0 and need_c and (v / z["px"] - 1 <= cur_sl):
                        z["exit"] = "be" if (cur_sl > -sl + 1e-9) else "stop"
                    elif tr > 0 and z["peak"] > z["px"] * (1 + act) and v <= z["peak"] * (1 - tr):
                        z["exit"] = "trail"

        # 5b) 0050 獨立盤中停損檢查 & 盤後閒置現金自動買入 0050
        if en_etf and etf_j is not None:
            adj_e = mk.cum.iat[i, etf_j] / etf_pos["cum"] if etf_pos["shares"] > 0 else 1.0
            if etf_pos["shares"] > 0 and etf_sl > 0 and i > etf_pos["hold_i"]:
                stop_epx = etf_pos["avg_px"] * (1.0 - etf_sl)
                low_e = float(mk.low.iat[i, etf_j]) / adj_e if mk.low.iat[i, etf_j] > 0 else float(mk.close_ff.iat[i, etf_j]) / adj_e
                open_e = float(mk.open.iat[i, etf_j]) / adj_e if mk.open.iat[i, etf_j] > 0 else low_e
                if low_e <= stop_epx:
                    exec_epx = min(open_e, stop_epx) * (1 - slip)
                    gross_e = exec_epx * etf_pos["shares"]
                    net_e = gross_e - max(20.0, gross_e * fee_rate) - gross_e * 0.001
                    pnl_e = net_e - etf_pos["cost"]
                    cash += net_e
                    stats["etf_sl_cnt"] += 1
                    stats["etf_pnl"] += pnl_e
                    etf_stopped_wait_reclaim = True
                    trades.append({
                        "side": "ETF", "stock_id": etf_code, "name": "元大台灣50", "reason": "Idle_ETF", "ym": str(d.date())[:7],
                        "YoY": np.nan, "YoY_Diff": np.nan, "MoM": np.nan, "MoM_Diff": np.nan,
                        "announcement_time": "", "ann_session": "idle", "entry_exec": "stop",
                        "buy_date": etf_pos["buy_date"], "buy_day_chg_pct": np.nan,
                        "buy_price": round(etf_pos["avg_px"], 2), "sell_date": d.date(), "sell_price": round(exec_epx, 2),
                        "lots": etf_pos["shares"] // 1000, "hold_days": i - etf_pos["hold_i"], "exit": "etf_stop_loss",
                        "peak_ret_pct": np.nan, "pnl": round(pnl_e, 0), "return_pct": round((net_e / etf_pos["cost"] - 1) * 100, 3)
                    })
                    etf_pos = {"shares": 0, "avg_px": 0.0, "cost": 0.0, "cum": 1.0, "buy_date": None, "hold_i": 0}

            # 收盤檢查是否符合 0050 均線進場條件，將多餘閒置現金投入 0050
            c_e = float(mk.close_ff.iat[i, etf_j])
            c_e_adj = float(etf_adj_close.iat[i])
            sma_e = float(etf_sma_series.iat[i]) if etf_sma_series is not None else np.nan
            sma_e_prev = float(etf_sma_series.iat[i - 1]) if (etf_sma_series is not None and i >= 1) else sma_e
            can_buy_etf = (not etf_stopped_wait_reclaim) and (etf_sma_n <= 0 or (c_e_adj == c_e_adj and sma_e == sma_e and c_e_adj > sma_e and sma_e >= sma_e_prev))
            if can_buy_etf and c_e > 0:
                cur_tot = cash + hv + (etf_pos["shares"] * (c_e / (mk.cum.iat[i, etf_j] / etf_pos["cum"])) if etf_pos["shares"] > 0 else 0.0)
                idle_cash = _avail_cash() - cur_tot * etf_buf
                buy_epx = c_e * (1 + slip)
                add_lots = int(idle_cash // (buy_epx * 1000))
                if add_lots >= 1:
                    add_sh = add_lots * 1000
                    add_cost = buy_epx * add_sh + max(20.0, buy_epx * add_sh * fee_rate)
                    if _avail_cash() >= add_cost:
                        cash -= add_cost
                        stats["etf_buy_cnt"] += 1
                        if etf_pos["shares"] == 0:
                            etf_pos = {"shares": add_sh, "avg_px": buy_epx, "cost": add_cost, "cum": mk.cum.iat[i, etf_j], "buy_date": d.date(), "hold_i": i}
                        else:
                            adj_cur = mk.cum.iat[i, etf_j] / etf_pos["cum"]
                            new_px_in_base = buy_epx / adj_cur
                            base_add_sh = add_sh * adj_cur
                            tot_base_sh = etf_pos["shares"] + base_add_sh
                            etf_pos["avg_px"] = (etf_pos["avg_px"] * etf_pos["shares"] + new_px_in_base * base_add_sh) / tot_base_sh
                            etf_pos["shares"] = tot_base_sh
                            etf_pos["cost"] += add_cost

        ev_f = (etf_pos["shares"] * (mk.close_ff.iat[i, etf_j] / (mk.cum.iat[i, etf_j] / etf_pos["cum"]))) if (en_etf and etf_j is not None and etf_pos["shares"] > 0) else 0.0
        sp_f = sum((sz["short_px"] - (mk.close_ff.iat[i, elig.cols[s]] / (mk.cum.iat[i, elig.cols[s]] / sz["cum"]))) * sz["shares"]
                   for s, sz in short_pos.items())
        equity.append((d, cash + hv + ev_f + sp_f, len(pos) + len(short_pos)))

    eq = pd.DataFrame(equity, columns=["date", "equity", "positions"]).set_index("date")
    
    return pd.DataFrame(trades), eq, len(pos) + len(short_pos), stats


def baseline_return(mk: Market, prep_rev: pd.DataFrame, elig: Eligibility, p: dict, bt: dict, start: pd.Timestamp) -> float:
    """對照基準：相同進場時點設定、相同市值/均量/產業濾網，不看營收條件、不停損，持有 N_Days 後開盤賣出的全市場平均每筆淨報酬(%)。"""
    fee_rate = 0.001425 * bt["Fee_Discount"]
    slip, N = bt["Slippage"], int(p["N_Days"])
    intra_proxy = str(p.get("Intraday_Price_Proxy", "mid")).lower()
    all_p = dict(p, X_YoY_Diff=-99999.0, Y_MoM_Diff=-99999.0, Min_YoY=-99999.0, Max_YoY=999999.0)
    all_sigs = make_daily_signals(prep_rev, mk.dates, all_p)
    if not all_sigs:
        return float("nan")

    rets = []
    for i, grp in all_sigs.items():
        if mk.dates[i] < start or i + N >= len(mk.dates):
            continue
        for r in grp.itertuples():
            sid, ex_type = r.stock_id, r.exec_type
            init_budget = target_budget(float(bt["Initial_Capital"]), p)
            if not elig.ok(i, sid, p, exec_type=ex_type, budget=init_budget):
                continue
            j = elig.cols[sid]
            raw_buy = _entry_raw_price(mk, i, j, ex_type, intra_proxy)
            o_sell = mk.open.iat[i + N, j]
            if not (raw_buy > 0 and o_sell > 0):
                continue
            buy = raw_buy * (1 + slip)
            sell = o_sell / (mk.cum.iat[i + N, j] / mk.cum.iat[i, j]) * (1 - slip)
            rets.append((sell * (1 - fee_rate - bt["Tax"])) / (buy * (1 + fee_rate)) - 1)
    return float(np.mean(rets) * 100) if rets else float("nan")


# =====================================================================
# 四、績效與風險指標
# =====================================================================
def metrics(trades: pd.DataFrame, eq: pd.DataFrame, unclosed: int, bt: dict) -> dict:
    m = {}
    if eq.empty:
        return m
    e = eq["equity"]
    years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
    m.update(
        final=e.iloc[-1],
        total_ret=(e.iloc[-1] / bt["Initial_Capital"] - 1) * 100,
        cagr=((e.iloc[-1] / bt["Initial_Capital"]) ** (1 / years) - 1) * 100,
        mdd=((e / e.cummax() - 1).min()) * 100,
        avg_pos=eq["positions"].mean(),
        unclosed=unclosed,
        n=len(trades),
    )
    dr = e.pct_change().dropna()
    m["sharpe"] = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else float("nan")
    stk_tr = trades[trades["side"] != "ETF"] if ("side" in trades.columns and len(trades)) else trades
    m["n"] = len(stk_tr)
    if len(stk_tr):
        w, l = stk_tr[stk_tr["pnl"] > 0], stk_tr[stk_tr["pnl"] <= 0]
        m.update(
            win=len(w) / len(stk_tr) * 100,
            avg_ret=stk_tr["return_pct"].mean(),
            med_ret=stk_tr["return_pct"].median(),
            avg_hold=stk_tr["hold_days"].mean(),
            pf=(w["pnl"].sum() / abs(l["pnl"].sum())) if len(l) and l["pnl"].sum() != 0 else float("inf"),
        )
    return m


def risk_metrics(eq: pd.DataFrame, m: dict, bt: dict, rf_pct: float) -> dict:
    out = {"calmar": np.nan, "ulcer": np.nan, "martin": np.nan, "recovery": np.nan, "mdd_amt": np.nan}
    if eq.empty or not m:
        return out
    e = eq["equity"]
    dd_pct = (e / e.cummax() - 1) * 100
    dd_amt = float((e.cummax() - e).max())
    ulcer = float(np.sqrt((dd_pct ** 2).mean()))
    mdd_abs = abs(dd_pct.min())
    out["ulcer"], out["mdd_amt"] = ulcer, dd_amt
    if mdd_abs > 0:
        out["calmar"] = m["cagr"] / mdd_abs
    if ulcer > 0:
        out["martin"] = (m["cagr"] - rf_pct) / ulcer
    if dd_amt > 0:
        out["recovery"] = (e.iloc[-1] - bt["Initial_Capital"]) / dd_amt
    return out


def fmt(v, nd=2):
    return "N/A" if v is None or v != v else f"{v:.{nd}f}"


# =====================================================================
# 五、圖表輸出
# =====================================================================
def _entry_desc(p: dict) -> str:
    pre = p.get("Pre_Market_Entry", "open")
    intra = p.get("Intraday_Entry", "immediate")
    proxy = p.get("Intraday_Price_Proxy", "mid")
    post = p.get("Post_Market_Entry", "next_open")
    pre_s = {"open": "當日開盤買", "close": "當日收盤買", "skip": "不買"}.get(pre, pre)
    intra_s = {"immediate": f"當下買({proxy})", "close": "當日收盤買", "next_open": "隔日開盤買", "skip": "不買"}.get(intra, intra)
    post_s = {"next_open": "隔日開盤買", "next_close": "隔日收盤買", "skip": "不買"}.get(post, post)
    return f"盤前(<{p['Pre_Market_End']})={pre_s}／盤中(<={p['Cutoff_Time']})={intra_s}／盤後={post_s}"


def param_lines(p: dict, bt: dict, start: pd.Timestamp, end: pd.Timestamp):
    sl, sl_mode = p["Stop_Loss_Pct"], str(p.get("Stop_Loss_Mode", "intraday")).lower()
    tr, act = p.get("Trail_Stop_Pct", 0.0), p.get("Trail_Activate_Pct", 0.0)
    min_yoy, max_yoy = p.get("Min_YoY"), p.get("Max_YoY")
    min_chg, min_ann = p.get("Min_Buy_Day_Chg"), int(p.get("Min_Ann_Day", 1))
    pos_pct = float(p.get("Position_Pct", 0.0))

    sl_str = (f"盤中-{sl:g}%停損" if sl_mode == "intraday" else f"收盤-{sl:g}%隔日賣") if sl > 0 else "關閉"
    d_cap = p.get("Dyn_SL_Cap_Pct")
    if sl > 0 and d_cap is not None and float(d_cap) > -999.0:
        sl_str += f"(動態上移至{float(d_cap):+g}%)"
    tr_str = f"獲利>{act:g}%回落>={tr:g}%賣" if tr > 0 else "關閉"
    has_min_y = (min_yoy is not None and float(min_yoy) > -999.0)
    has_max_y = (max_yoy is not None and float(max_yoy) < 99999.0)
    if has_min_y and has_max_y:
        yoy_str = f"{float(min_yoy):g}%<當月YoY<={float(max_yoy):g}%"
    elif has_min_y:
        yoy_str = f"當月YoY>{float(min_yoy):g}%"
    elif has_max_y:
        yoy_str = f"當月YoY<={float(max_yoy):g}%"
    else:
        yoy_str = "不限YoY"
    chg_str = f"漲跌>={float(min_chg):+g}%" if (min_chg is not None and float(min_chg) > -50.0) else "不限漲跌"
    max_chg = p.get("Max_Buy_Day_Chg")
    if max_chg is not None and float(max_chg) < 50.0:
        chg_str += f" & <={float(max_chg):+g}%"
    max_ru = p.get("Max_Pre_Runup_Pct")
    if max_ru is not None and float(max_ru) < 999.0:
        chg_str += f"（前{int(p.get('Pre_Runup_Days', 3))}日累漲<={float(max_ru):g}%）"
    ann_str = f"公告日>={min_ann}" if min_ann > 1 else "不限公告日"
    alloc_str = f"動態總資產 {pos_pct:g}%/檔" if pos_pct > 0 else f"固定 {p['Position_Size']:,.0f}/檔"
    gs_str = (f"跳空{p['Gap_Short_Min_Pct']:g}~{p['Gap_Short_Max_Pct']:g}%放空(停損{p['Short_Stop_Loss_Pct']:g}%/正常回補反手作多)"
              if p.get("Enable_Gap_Short", True) else "關閉")

    eps_parts = []
    if p.get("Enable_EPS_Filter", False):
        if float(p.get("Min_Last_Q_EPS", -999.0)) > -999.0:
            eps_parts.append(f"單季EPS>{float(p['Min_Last_Q_EPS']):g}")
        if float(p.get("Min_TTM_EPS", -999.0)) > -999.0:
            eps_parts.append(f"近4季EPS>{float(p['Min_TTM_EPS']):g}")
        if float(p.get("Min_Q_EPS_YoY", -999.0)) > -999.0:
            eps_parts.append(f"EPS年增>{float(p['Min_Q_EPS_YoY']):g}%")
        if float(p.get("Max_PE_Ratio", 999.0)) < 999.0:
            eps_parts.append(f"PE<={float(p['Max_PE_Ratio']):g}")
    eps_str = ("　|　EPS：" + " & ".join(eps_parts)) if eps_parts else ""
    l1 = f"進場：[{yoy_str}] 且 [(YoY差)>{p['X_YoY_Diff']:g}% 或 連3月(MoM差)>{p['Y_MoM_Diff']:g}%]{eps_str}　|　{chg_str}　|　{ann_str}　|　功能2：{gs_str}"
    sz_m = str(p.get("Size_Mode", "adj_cap")).lower()
    sz_lbl = "成交額" if sz_m == "turnover" else ("還原市值" if sz_m == "adj_cap" else "靜態市值")
    to_v = float(p.get("Min_Avg_Turnover", 0))
    to_pct = float(p.get("Max_Turnover_Pct", 0))
    to_parts = []
    if to_v > 0:
        to_parts.append(f"均額>={to_v / 1e4:g}萬")
    if to_pct > 0:
        to_parts.append(f"均額>=投入/{to_pct:g}")
    to_lbl = (" & " + " & ".join(to_parts)) if to_parts else ""
    l2 = (f"出場：持有{p['N_Days']}日開盤賣　|　停損：{sl_str}　|　移動停利：{tr_str}　|　"
          f"{sz_lbl}D{p['Cap_Decile_Low']}~D{p['Cap_Decile_High']}、{p['Vol_Window']}日均量>={p['Min_Avg_Volume']:g}張{to_lbl}")
    l3 = f"期間：{start.date()} ~ {end.date()}　|　時機：{_entry_desc(p)}　|　本金 {bt['Initial_Capital']:,.0f}、{alloc_str}、上限 {p['Max_Positions']} 檔"
    return [l1, l2, l3]


def monthly_growth(eq: pd.DataFrame, initial: float) -> pd.Series:
    me = eq["equity"].resample("ME").last()
    prev = me.shift(1)
    prev.iloc[0] = initial
    g = (me / prev - 1) * 100
    g.index = g.index.strftime("%Y-%m")
    return g


def draw_chart(path, p, bt, start, end, eq, trades, m, rk=None, rf=RF_RATE, stats=None):
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.6, 2, 1.5], hspace=0.38, top=0.83, bottom=0.05, left=0.07, right=0.97)
    lines = param_lines(p, bt, start, end)
    fig.text(0.07, 0.975, "月營收動能策略回測 new_v16（修復漲停鎖死成交＋極端基期YoY上限過濾＋減資異常保護）（動態總資產比例資金控管 ＋ 開盤跳空過熱放空反手策略）", fontsize=14.0, fontweight="bold", va="top")
    for k, s in enumerate(lines):
        fig.text(0.07, 0.945 - k * 0.02, s, fontsize=8.9, va="top", color="#222")
    if m and m.get("n"):
        ex = trades["exit"].value_counts().to_dict()
        n_stop = ex.get("stop", 0) + ex.get("stop_intraday", 0) + ex.get("stop_gap", 0)
        n_be = ex.get("be", 0) + ex.get("be_intraday", 0) + ex.get("be_gap", 0)
        gs_s = (f"　|　跳空放空{stats['gap_short_cnt']}筆(停損{stats['gap_short_sl']}/漲停隔日補{stats['gap_short_limit_next']}/反手作多{stats['gap_short_reverse_long']})"
                if stats and stats.get("gap_short_cnt", 0) > 0 else "")
        skip_str = (f"　|　漲跌跳過{stats['skip_buy_chg']}筆／防追高跳過{stats.get('skip_chase', 0)}筆{gs_s}" if stats else "")
        summ = (f"總報酬 {m['total_ret']:+.1f}%　年化 {m['cagr']:+.1f}%　最大回撤 {m['mdd']:.1f}%　Sharpe {m['sharpe']:.2f}　"
                f"交易 {m['n']} 筆　勝率 {m['win']:.1f}%　平均 {m['avg_ret']:+.2f}%/筆（獲利因子 {m['pf']:.2f}）\n"
                f"Calmar {fmt(rk['calmar'])}　Martin {fmt(rk['martin'])}（無風險 {rf:g}%、Ulcer {fmt(rk['ulcer'])}）　"
                f"恢復因子 {fmt(rk['recovery'])}　|　出場：到期 {ex.get('time', 0)}／停損 {n_stop}(盤中{ex.get('stop_intraday', 0)}/跳空{ex.get('stop_gap', 0)}/收盤{ex.get('stop', 0)})／動態停損 {n_be}／移動停利 {ex.get('trail', 0)}{skip_str}")
    else:
        summ = "（期間內無交易）"
    fig.text(0.07, 0.882, summ, fontsize=9.8, fontweight="bold", color="#0b5394", va="top")

    e = eq["equity"]
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(e.index, e / 1e4, color="#1f4e79", lw=1.6)
    ax1.axhline(bt["Initial_Capital"] / 1e4, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("總資產（萬元）")
    ax1.set_title("資產曲線", loc="left", fontsize=10)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    dd = (e / e.cummax() - 1) * 100
    ax2.fill_between(dd.index, dd, 0, color="#cc4125", alpha=0.45)
    imin = dd.idxmin()
    ax2.annotate(f"最大回撤 {dd.min():.2f}%\n{imin.date()}", xy=(imin, dd.min()), xytext=(10, 8),
                 textcoords="offset points", fontsize=9, color="#7f1d0d",
                 arrowprops=dict(arrowstyle="-", color="#7f1d0d"))
    ax2.set_ylabel("回撤 %")
    ax2.set_title("回撤", loc="left", fontsize=10)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    g = monthly_growth(eq, bt["Initial_Capital"])
    ax3.bar(g.index, g.values, color=np.where(g.values >= 0, "#2e8b57", "#cc4125"))
    if len(g) <= 36:
        for x, v in zip(g.index, g.values):
            ax3.annotate(f"{v:+.1f}%", (x, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8,
                         xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
    ax3.axhline(0, color="black", lw=0.7)
    ax3.set_ylabel("月成長率 %")
    ax3.set_title("每月成長率（以月底資產 / 上月底資產）", loc="left", fontsize=10)
    ax3.margins(y=0.25)
    ax3.grid(axis="y", alpha=0.3)
    if len(g) > 36:
        step = max(1, len(g) // 24)
        for idx_lbl, lbl in enumerate(ax3.get_xticklabels()):
            if idx_lbl % step != 0:
                lbl.set_visible(False)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.fill_between(eq.index, eq["positions"], step="post", color="#6aa84f", alpha=0.6)
    ax4.axhline(p["Max_Positions"], color="gray", ls=":", lw=0.8)
    ax4.set_ylabel("持倉檔數")
    ax4.set_title(f"同時持有倉數（上限 {p['Max_Positions']}，平均 {eq['positions'].mean():.1f}）", loc="left", fontsize=10)
    ax4.set_ylim(0, max(p["Max_Positions"], int(eq["positions"].max()) if not eq.empty else 1) + 1)
    ax4.grid(alpha=0.3)

    fig.savefig(path, dpi=110)
    plt.close(fig)


def tag_of(p):
    tr = p.get("Trail_Stop_Pct", 0.0)
    act = p.get("Trail_Activate_Pct", 0.0)
    tr_tag = f"_TR{tr:g}A{act:g}" if tr > 0 else ""
    gs_tag = "_GS" if p.get("Enable_Gap_Short", False) else ""
    d_cap = p.get("Dyn_SL_Cap_Pct")
    dyn_tag = f"_Dyn{float(d_cap):+g}" if (d_cap is not None and float(d_cap) > -999.0) else ""
    sl_mode = "I" if str(p.get("Stop_Loss_Mode", "intraday")).lower() == "intraday" else "C"
    min_yoy, max_yoy = p.get("Min_YoY"), p.get("Max_YoY")
    yoy_tag = f"_MinYoY{float(min_yoy):g}" if (min_yoy is not None and float(min_yoy) > -999.0) else "_MinYoYOff"
    if max_yoy is not None and float(max_yoy) < 99999.0:
        yoy_tag += f"_MaxYoY{float(max_yoy):g}"
    min_chg = p.get("Min_Buy_Day_Chg")
    chg_tag = f"_Chg{float(min_chg):+g}" if (min_chg is not None and float(min_chg) > -50.0) else "_ChgOff"
    max_chg = p.get("Max_Buy_Day_Chg")
    mchg_tag = f"_MaxC{float(max_chg):g}" if (max_chg is not None and float(max_chg) < 50.0) else ""
    max_ru = p.get("Max_Pre_Runup_Pct")
    ru_tag = f"_RU{int(p.get('Pre_Runup_Days', 3))}d{float(max_ru):g}" if (max_ru is not None and float(max_ru) < 999.0) else ""
    min_ann = int(p.get("Min_Ann_Day", 1))
    ann_tag = f"_Ann{min_ann}" if min_ann > 1 else ""
    e_tag = f"_{p['Pre_Market_Entry'][0].upper()}{p['Intraday_Entry'][0].upper()}{p['Post_Market_Entry'].replace('next_', 'n')[0:2].upper()}"
    sz_m = str(p.get("Size_Mode", "adj_cap")).lower()
    sz_pfx = "T" if sz_m == "turnover" else ("AC" if sz_m == "adj_cap" else "C")
    pos_tag = f"_M{p['Max_Positions']}P{p['Position_Pct']:.2f}"
    eps_tag = ""
    if p.get("Enable_EPS_Filter", False):
        if float(p.get("Min_Last_Q_EPS", -999.0)) > -999.0:
            eps_tag += f"_QEPS{float(p['Min_Last_Q_EPS']):g}"
        if float(p.get("Min_TTM_EPS", -999.0)) > -999.0:
            eps_tag += f"_TTM{float(p['Min_TTM_EPS']):g}"
        if float(p.get("Min_Q_EPS_YoY", -999.0)) > -999.0:
            eps_tag += f"_EPSY{float(p['Min_Q_EPS_YoY']):g}"
        if float(p.get("Max_PE_Ratio", 999.0)) < 999.0:
            eps_tag += f"_PE{float(p['Max_PE_Ratio']):g}"
    return (f"dX{p['X_YoY_Diff']:g}_dY{p['Y_MoM_Diff']:g}{yoy_tag}{eps_tag}_N{p['N_Days']}_SL{p['Stop_Loss_Pct']:g}{sl_mode}{dyn_tag}{tr_tag}{gs_tag}"
            f"{e_tag}{chg_tag}{mchg_tag}{ru_tag}{ann_tag}{pos_tag}_{sz_pfx}D{p['Cap_Decile_Low']}-{p['Cap_Decile_High']}")


def parse_list(s, cast):
    return [cast(x) for x in str(s).split(",") if x.strip()]


# =====================================================================
# 六、主程式
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="台股月營收動能策略回測 new_v16（當月YoY底線 + YoY月差值 & 連3月MoM月差值 + 三時段進場）")
    ap.add_argument("--start", default="2018-01-01", help="回測起始日 YYYY-MM-DD（預設 2015-01-01）")
    ap.add_argument("--end", default="2023-01-01", help="回測起始日 YYYY-MM-DD（預設 2015-01-01）")
    # ap.add_argument("--end", default=dt.date.today().isoformat(), help="回測結束日 YYYY-MM-DD")
    ap.add_argument("--download", action="store_true", help="下載缺少的每日行情並將舊 CSV 升級補齊 high/low 欄位（可中斷續傳）")
    ap.add_argument("--x", type=float, help="當月 YoY 減 上月 YoY 差值門檻 %%（X_YoY_Diff）")
    ap.add_argument("--y", type=float, help="連 3 月（MoM 減 上月 MoM）差值門檻 %%（Y_MoM_Diff）")
    ap.add_argument("--min-yoy", type=float, help="v5新增：當月 YoY 必須大於此底線 %%（Min_YoY，預設 0；設 -999 關閉）")
    ap.add_argument("--download-eps", action="store_true", help="v16新增：下載/更新季報 EPS 快取")
    ap.add_argument("--no-eps", action="store_true", help="v16新增：關閉季報 EPS 濾網")
    ap.add_argument("--min-q-eps", type=float, help="v16新增：最新單季 EPS 下限（設 -999 關閉）")
    ap.add_argument("--min-ttm-eps", type=float, help="v16新增：近四季累計 EPS 下限（設 -999 關閉）")
    ap.add_argument("--min-eps-yoy", type=float, help="v16新增：最新單季 EPS 年增率 %% 下限（設 -999 關閉）")
    ap.add_argument("--max-pe", type=float, help="v16新增：本益比上限（設 999 關閉）")
    ap.add_argument("--max-yoy", type=float, help="v14新增：當月 YoY 上限 %%（Max_YoY，預設 500；設 99999 關閉）")
    ap.add_argument("--n", type=int, help="持有交易日數 N")
    ap.add_argument("--stop-loss", type=float, help="停損 %%（0=不停損）")
    ap.add_argument("--dyn-cap", type=float, help="動態上移停損上限 X%%（每漲1%%停損上移1%%，最多拉高至此%%；設 -999 關閉）")
    ap.add_argument("--sl-mode", choices=["intraday", "close"], help="停損模式：intraday=盤中觸價停損(預設)、close=收盤停損隔日開盤賣")
    ap.add_argument("--trail-stop", type=float, help="移動停利回落 %%（0=關閉）")
    ap.add_argument("--trail-activate", type=float, help="移動停利啟動門檻 %%（預設 0.0）")

    # 三時段進場時機 CLI 參數
    ap.add_argument("--pre-end", default=None, help="盤前公告結束時間 HH:MM:SS（預設 09:00:00）")
    ap.add_argument("--cutoff-time", default=None, help="盤中/盤後公告分界時間 HH:MM:SS（預設 13:25:00）")
    ap.add_argument("--pre-entry", choices=["open", "close", "skip"],
                    help="盤前公告買進時機：open=當天開盤即買入 | close=當天收盤買入 | skip=不買")
    ap.add_argument("--intra-entry", choices=["immediate", "close", "next_open", "skip"],
                    help="盤中公告買進時機：immediate=公告當下即買進 | close=當日收盤買入 | next_open=隔日開盤買 | skip=不買")
    ap.add_argument("--intra-proxy", choices=["mid", "hl2", "close", "open"],
                    help="盤中當下買進之日K價格代理：mid=(開+收)/2 | hl2=(高+低)/2 | close=收盤 | open=開盤")
    ap.add_argument("--post-entry", choices=["next_open", "next_close", "skip"],
                    help="盤後公告買進時機：next_open=隔天開盤即買入 | next_close=隔天收盤買入 | skip=不買")

    ap.add_argument("--min-buy-chg", type=float, help="買進時點個股最低漲跌幅 %%（設 -99 關閉）")
    ap.add_argument("--max-buy-chg", type=float, help="防追高1：買進時點當日最高漲幅上限 %%（設 99 關閉）")
    ap.add_argument("--runup-days", type=int, help="防追高2：進場前累積漲幅觀察天數（預設 3）")
    ap.add_argument("--max-runup", type=float, help="防追高2：進場前 N 日累積漲幅上限 %%（設 999 關閉）")
    ap.add_argument("--min-ann-day", type=int, help="營收公告日下限（預設 1 不過濾；設 6 排除每月 1~5 日早鳥公告）")
    ap.add_argument("--size-mode", choices=["adj_cap", "cap", "turnover"], help="規模分位基準：adj_cap=還原歷史股本市值(預設)、cap=靜態股本市值、turnover=均成交金額")
    ap.add_argument("--cap-low", type=int, help="市值十分位下界 1~10")
    ap.add_argument("--cap-high", type=int, help="市值十分位上界 1~10")
    ap.add_argument("--min-vol", type=float, help="近 20 日均量下限（張）")
    ap.add_argument("--min-turnover", type=float, help="固定均成交金額保底下限（元，預設 0 關閉）")
    ap.add_argument("--max-to-pct", type=float, help="單筆投入預算占個股20日均成交額上限比例（例 0.03 代表 3%%；設 0 關閉）")
    ap.add_argument("--capital", type=float, help="初始本金")
    ap.add_argument("--pos-pct", type=float, help="每檔投入動態總資產 X%%（預設 2.0）")
    ap.add_argument("--position-size", type=float, help="備用固定金額上限")
    ap.add_argument("--gap-short", action="store_true", help="開啟開盤跳空放空反手（功能2）")
    ap.add_argument("--no-gap-short", action="store_true", help="關閉開盤跳空放空")
    ap.add_argument("--gs-min", type=float, help="跳空放空下限 X%%（預設 3.0）")
    ap.add_argument("--gs-max", type=float, help="跳空放空上限 Y%%（預設 8.5）")
    ap.add_argument("--short-sl", type=float, help="放空盤中停損 %%（預設 4.0）")
    ap.add_argument("--max-pos", type=int, help="同時最多持有檔數")
    ap.add_argument("--rf", type=float, default=RF_RATE, help="無風險利率 %%/年（Martin Ratio 用，預設 1.5）")

    # Grid 模式參數（沿用使用者在 v4 的預設掃描清單 + 支援 --grid-min-yoy）
    ap.add_argument("--grid", default=False, action="store_true", help="執行參數掃描，每組輸出一張圖與 grid_summary.csv")
    ap.add_argument("--single", default=False, action="store_true", help="強制只跑單次回測（不跑 Grid）")
    ap.add_argument("--grid-x", default="10,20,30,40,50,60", help="YoY_Diff 差值門檻掃描清單")
    ap.add_argument("--grid-y", default="2,4,6,8,10,12", help="連3月 MoM_Diff 差值門檻掃描清單")
    ap.add_argument("--grid-min-yoy", default="0", help="v5當月 YoY 底線掃描清單，例 -999,0,10,20（預設使用 PARAMS['Min_YoY']）")
    ap.add_argument("--grid-q-eps", default=None, help="v16單季 EPS 下限掃描清單，例 -999,0,0.5,1.0")
    ap.add_argument("--grid-ttm-eps", default=None, help="v16近四季 EPS 下限掃描清單，例 -999,0,2.0")
    ap.add_argument("--grid-max-pe", default=None, help="v16本益比上限掃描清單，例 999,20,30")
    ap.add_argument("--grid-n", default=",5,10,15,20", help="持有天數 N 掃描清單")
    ap.add_argument("--grid-sl", default="0", help="停損 %% 掃描清單")
    ap.add_argument("--grid-dyn-cap", default="-999", help="動態上移停損上限 X%% 掃描清單（-999=關閉）")
    ap.add_argument("--grid-trail", default="0", help="移動停利 %% 掃描清單")
    ap.add_argument("--grid-trail-act", default="0", help="移動停利啟動門檻 %% 掃描清單")
    ap.add_argument("--grid-gap-short", default="0", help="功能2跳空放空掃描清單（0=關, 1=開）")
    ap.add_argument("--grid-buy-chg", default="-99", help="買進時點最低漲跌幅 %% 掃描清單，例 -99,0,1")
    ap.add_argument("--grid-max-chg", default="99", help="防追高1：當日最高漲幅上限 %% 掃描清單（99=關閉）")
    ap.add_argument("--grid-max-runup", default="999", help="防追高2：前N日累積漲幅上限 %% 掃描清單（999=關閉）")
    ap.add_argument("--grid-post-entry", default=None, help="盤後公告買進時機掃描清單，例 next_open,next_close")
    ap.add_argument("--grid-intra-entry", default=None, help="盤中公告買進時機掃描清單，例 immediate,close")
    ap.add_argument("--grid-max-pos", default="20,30,40,50,60", help="同時最多持有檔數掃描清單（Position_Pct 自動設為 100/Max_Positions），例 20,30,40,50,60")
    ap.add_argument("--max-combos", type=int, default=200)
    ap.add_argument("--force", default=True, action="store_true", help="組合數超過 --max-combos 時強制執行")
    a = ap.parse_args()

    p, bt = dict(PARAMS), dict(BT)
    for k, v in [
        ("X_YoY_Diff", a.x), ("Y_MoM_Diff", a.y), ("Min_YoY", a.min_yoy),
        ("Min_Last_Q_EPS", a.min_q_eps), ("Min_TTM_EPS", a.min_ttm_eps),
        ("Min_Q_EPS_YoY", a.min_eps_yoy), ("Max_PE_Ratio", a.max_pe), ("Max_YoY", a.max_yoy), ("N_Days", a.n),
        ("Stop_Loss_Pct", a.stop_loss), ("Dyn_SL_Cap_Pct", a.dyn_cap), ("Stop_Loss_Mode", a.sl_mode),
        ("Trail_Stop_Pct", a.trail_stop), ("Trail_Activate_Pct", a.trail_activate),
        ("Pre_Market_End", a.pre_end), ("Cutoff_Time", a.cutoff_time),
        ("Pre_Market_Entry", a.pre_entry), ("Intraday_Entry", a.intra_entry),
        ("Intraday_Price_Proxy", a.intra_proxy), ("Post_Market_Entry", a.post_entry),
        ("Min_Buy_Day_Chg", a.min_buy_chg), ("Max_Buy_Day_Chg", a.max_buy_chg), ("Pre_Runup_Days", a.runup_days), ("Max_Pre_Runup_Pct", a.max_runup), ("Min_Ann_Day", a.min_ann_day),
        ("Cap_Decile_Low", a.cap_low), ("Cap_Decile_High", a.cap_high),
        ("Size_Mode", a.size_mode), ("Min_Avg_Volume", a.min_vol), ("Min_Avg_Turnover", a.min_turnover), ("Max_Turnover_Pct", a.max_to_pct), ("Position_Pct", a.pos_pct), ("Position_Size", a.position_size), ("Max_Positions", a.max_pos),
        ("Gap_Short_Min_Pct", a.gs_min), ("Gap_Short_Max_Pct", a.gs_max), ("Short_Stop_Loss_Pct", a.short_sl),
    ]:
        if v is not None:
            p[k] = v
    if a.gap_short:
        p["Enable_Gap_Short"] = True
    if a.no_gap_short:
        p["Enable_Gap_Short"] = False
    if a.no_eps:
        p["Enable_EPS_Filter"] = False
    if a.capital:
        bt["Initial_Capital"] = a.capital
    if not (1 <= p["Cap_Decile_Low"] <= p["Cap_Decile_High"] <= 10):
        sys.exit(f"❌ 市值十分位需滿足 1 <= Cap_Decile_Low({p['Cap_Decile_Low']}) <= Cap_Decile_High({p['Cap_Decile_High']}) <= 10")

    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    warm = (start - pd.Timedelta(days=45)).date()
    s_ym = add_months(start.year, start.month, -5)
    e_ym = (end.year, end.month)

    if a.download:
        update_price_cache(warm, end.date())

    prices, missing, no_hl_days = load_prices(warm, end.date())
    if p.get("Enable_Idle_ETF", False):
        prices = ensure_etf_prices(prices, str(p.get("Idle_ETF_Code", "0050")), warm, end.date())
    if prices.empty:
        sys.exit("❌ data_cache/prices 沒有行情資料，請加 --download 執行。")
    if missing:
        print(f"⚠️ 有 {len(missing)} 個平日行情尚未下載（首個 {missing[0]}），結果可能失真；可加 --download 補齊。")
    if no_hl_days > 0 and str(p.get("Stop_Loss_Mode", "intraday")).lower() == "intraday" and p["Stop_Loss_Pct"] > 0:
        print(f"ℹ️ 目前有 {no_hl_days} 個交易日的舊版快取尚未包含 high/low 欄位（目前對該日期僅能做開盤跳空停損 + 收盤停損）；"
              f"請執行 `python backtest_new_v16.py --download` 補齊盤中最高/最低價。")

    industry_map = load_industry_map()
    rev_all = load_ext_revenue(s_ym, e_ym, industry_map)
    if rev_all.empty:
        sys.exit(f"❌ {EXT_REV_DIR} 找不到月營收資料，請先執行 external_lib/batch_run.py。")

    if a.download_eps:
        update_eps_cache(start.year - 2, end.year)
    shares_map = load_shares()
    mk = Market(prices, p["Vol_Window"])
    eps_df = load_and_prepare_eps(mk.dates)
    if p.get("Enable_EPS_Filter", False) and eps_df.empty:
        print("data_cache/eps 尚無季報 EPS 資料（可加 --download-eps 下載，或放入 CSV），本次自動略過 EPS 濾網。")
    elif not eps_df.empty:
        print(f"季報 EPS 已載入：共 {len(eps_df):,} 筆季報紀錄（涵蓋 {eps_df['stock_id'].nunique()} 檔股票）")
    elig = Eligibility(mk, shares_map, industry_map, eps_df=eps_df)
    prep_rev = prepare_revenue_table(rev_all, mk.dates)

    total_rev = len(prep_rev)
    valid_ann = int(prep_rev["valid_ann"].sum())
    print(f"✅ 行情 {len(mk.dates)} 個交易日 / {mk.close.shape[1]} 檔；"
          f"月營收 {rev_all['ym'].nunique()} 個月（共 {total_rev:,} 筆，具精確申報時間 {valid_ann:,} 筆 / {valid_ann / max(1, total_rev) * 100:.1f}%）")

    run_dir = os.path.join(OUT_ROOT, "new_v16_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # ---------------- Grid 參數掃描 ----------------
    if a.grid and not a.single:
        one = lambda key, raw, cast: parse_list(raw, cast) if raw is not None else [p[key]]  # noqa: E731
        any_custom = any(x is not None for x in (
            a.grid_x, a.grid_y, a.grid_min_yoy, a.grid_q_eps, a.grid_ttm_eps, a.grid_max_pe, a.grid_n, a.grid_sl, a.grid_dyn_cap, a.grid_trail, a.grid_trail_act, a.grid_gap_short, a.grid_buy_chg, a.grid_max_chg, a.grid_max_runup, a.grid_post_entry, a.grid_intra_entry, a.grid_max_pos
        ))
        xs = one("X_YoY_Diff", a.grid_x, float)
        ys = one("Y_MoM_Diff", a.grid_y, float)
        min_yoys = one("Min_YoY", a.grid_min_yoy, float)
        q_eps_list = one("Min_Last_Q_EPS", a.grid_q_eps, float)
        ttm_eps_list = one("Min_TTM_EPS", a.grid_ttm_eps, float)
        max_pe_list = one("Max_PE_Ratio", a.grid_max_pe, float)
        ns = one("N_Days", a.grid_n, int)
        sls = one("Stop_Loss_Pct", a.grid_sl, float) if any_custom else [0.0, 8.0, 10.0]
        dyn_caps = one("Dyn_SL_Cap_Pct", a.grid_dyn_cap, float)
        trs = one("Trail_Stop_Pct", a.grid_trail, float)
        tr_acts = one("Trail_Activate_Pct", a.grid_trail_act, float)
        gs_list = [bool(int(x)) for x in str(a.grid_gap_short).split(",") if x.strip()] if a.grid_gap_short is not None else [p["Enable_Gap_Short"]]
        chgs = one("Min_Buy_Day_Chg", a.grid_buy_chg, float) if any_custom else [-99.0, 0.0]
        max_chgs = one("Max_Buy_Day_Chg", a.grid_max_chg, float)
        max_runups = one("Max_Pre_Runup_Pct", a.grid_max_runup, float)
        post_es = one("Post_Market_Entry", a.grid_post_entry, str) if any_custom else [p["Post_Market_Entry"]]
        intra_es = one("Intraday_Entry", a.grid_intra_entry, str) if any_custom else [p["Intraday_Entry"]]
        max_pos_list = one("Max_Positions", a.grid_max_pos, int)

        raw_combos = list(itertools.product(xs, ys, min_yoys, q_eps_list, ttm_eps_list, max_pe_list, ns, sls, dyn_caps, trs, tr_acts, gs_list, chgs, max_chgs, max_runups, post_es, intra_es, max_pos_list))
        combos = [c for c in raw_combos if not (c[9] == 0.0 and c[10] != tr_acts[0])]
        if len(combos) > a.max_combos and not a.force:
            sys.exit(f"❌ 組合數 {len(combos)} 超過上限 {a.max_combos}（每組一張圖），請縮小清單或加 --force。")
        print(f"🔁 Grid 共 {len(combos)} 種組合，每種輸出一張圖 → {run_dir}")

        rows, sig_cache = [], {}
        for k, (X, Y, MIN_YOY, Q_EPS, TTM_EPS, MAX_PE, N, SL, DYN_CAP, TR, TR_ACT, GS_ON, CHG, MAX_CHG, MAX_RU, POST_E, INTRA_E, MAX_POS) in enumerate(combos, 1):
            PRE_E = "open" if POST_E == "next_open" else p["Pre_Market_Entry"]
            POS_PCT = round(100.0 / MAX_POS, 4) if a.grid_max_pos is not None else p["Position_Pct"]
            q = dict(p, X_YoY_Diff=X, Y_MoM_Diff=Y, Min_YoY=MIN_YOY, Min_Last_Q_EPS=Q_EPS, Min_TTM_EPS=TTM_EPS, Max_PE_Ratio=MAX_PE, N_Days=N, Stop_Loss_Pct=SL, Dyn_SL_Cap_Pct=DYN_CAP,
                     Trail_Stop_Pct=TR, Trail_Activate_Pct=TR_ACT, Enable_Gap_Short=GS_ON,
                     Min_Buy_Day_Chg=CHG, Max_Buy_Day_Chg=MAX_CHG, Max_Pre_Runup_Pct=MAX_RU, Post_Market_Entry=POST_E, Intraday_Entry=INTRA_E, Pre_Market_Entry=PRE_E,
                     Max_Positions=MAX_POS, Position_Pct=POS_PCT)
            ck = (X, Y, MIN_YOY, q["Min_Ann_Day"], q["Pre_Market_Entry"], q["Intraday_Entry"], q["Post_Market_Entry"])
            if ck not in sig_cache:
                sig_cache[ck] = make_daily_signals(prep_rev, mk.dates, q)
            tr_df, eq, unc, st = simulate(mk, sig_cache[ck], elig, q, bt, start)
            m = metrics(tr_df, eq, unc, bt)
            rk = risk_metrics(eq, m, bt, a.rf)
            fname = f"{k:03d}_{tag_of(q)}.png"
            draw_chart(os.path.join(run_dir, fname), q, bt, start, end, eq, tr_df, m, rk, a.rf, st)
            ex = tr_df["exit"].value_counts().to_dict() if len(tr_df) else {}
            n_stop = ex.get("stop", 0) + ex.get("stop_intraday", 0) + ex.get("stop_gap", 0)
            n_be = ex.get("be", 0) + ex.get("be_intraday", 0) + ex.get("be_gap", 0)
            rows.append({
                "編號": k,
                "上限檔數": MAX_POS,
                "單筆比例%": round(POS_PCT, 2),
                "當月YoY>": "不限" if MIN_YOY <= -999 else f"{MIN_YOY:g}%",
                "單季EPS>": "不限" if Q_EPS <= -999 else f"{Q_EPS:g}",
                "近4季EPS>": "不限" if TTM_EPS <= -999 else f"{TTM_EPS:g}",
                "本益比<=": "不限" if MAX_PE >= 999 else f"{MAX_PE:g}",
                "dX(YoY差)": X, "dY(MoM差)": Y, "N": N, "停損%": SL, "動態停損上限%": ("關閉" if DYN_CAP <= -999 else f"{DYN_CAP:+g}%"), "啟動門檻%": (TR_ACT if TR > 0 else 0.0), "移動停利%": TR, "功能2": ("開" if GS_ON else "關"),
                "盤前": PRE_E, "盤中": INTRA_E, "盤後": POST_E,
                "漲跌確認": "關閉" if CHG <= -50 else f">={CHG:+g}%",
                "當日漲幅上限%": ("關閉" if MAX_CHG >= 50 else f"<={MAX_CHG:g}%"),
                "前N日累漲上限%": ("關閉" if MAX_RU >= 999 else f"<={MAX_RU:g}%"),
                "筆數": m.get("n", 0),
                "勝率%": round(m.get("win", np.nan), 1),
                "平均%": round(m.get("avg_ret", np.nan), 2),
                "獲利因子": round(m.get("pf", np.nan), 2),
                "總報酬%": round(m.get("total_ret", np.nan), 1),
                "年化%": round(m.get("cagr", np.nan), 2),
                "MDD%": round(m.get("mdd", np.nan), 2),
                "Sharpe": round(m.get("sharpe", np.nan), 2),
                "Calmar": round(rk["calmar"], 2),
                "Martin": round(rk["martin"], 2),
                "恢復因子": round(rk["recovery"], 2),
                "停損筆": n_stop,
                "動態停損筆": n_be,
                "圖檔": fname,
            })
            if k % 12 == 0 or k == len(combos):
                print(f"   進度 {k}/{len(combos)}")

        g = pd.DataFrame(rows)
        g.to_csv(os.path.join(run_dir, "grid_summary.csv"), index=False, encoding="utf-8-sig")
        print("\n📊 Grid 掃描結果（依 Calmar 由高到低前 20 名）：")
        print(g.sort_values("Calmar", ascending=False).head(20).drop(columns=["圖檔"]).to_string(index=False))
        print(f"\n💾 {len(combos)} 張圖與 grid_summary.csv 已輸出：{run_dir}")
        print("⚠️ 掃描最佳值容易過度擬合，請以更長歷史或樣本外期間再驗證。")
        return

    # ---------------- 單次回測 ----------------
    sigs = make_daily_signals(prep_rev, mk.dates, p)
    trades, eq, unc, st = simulate(mk, sigs, elig, p, bt, start)
    m = metrics(trades, eq, unc, bt)
    rk = risk_metrics(eq, m, bt, a.rf)
    base = baseline_return(mk, prep_rev, elig, p, bt, start)

    print("\n" + "=" * 88)
    for s in param_lines(p, bt, start, end):
        print(s)
    print("-" * 88)
    if m.get("n"):
        sess_cnt = trades["ann_session"].value_counts().to_dict()
        exec_cnt = trades["entry_exec"].value_counts().to_dict()
        rsn_cnt = trades["reason"].value_counts().to_dict()
        ex = trades["exit"].value_counts().to_dict()
        print(f"期末資產 {m['final']:,.0f}｜總報酬 {m['total_ret']:.2f}%｜年化 {m['cagr']:.2f}%｜"
              f"MDD {m['mdd']:.2f}%｜Sharpe {m['sharpe']:.2f}")
        print(f"交易 {m['n']} 筆｜觸發原因：{rsn_cnt}")
        print(f"公告時段分佈：盤前 {sess_cnt.get('pre_market', 0)} 筆／盤中 {sess_cnt.get('intraday', 0)} 筆／盤後 {sess_cnt.get('post_market', 0)} 筆")
        print(f"實際執行買進時點：開盤買進(open) {exec_cnt.get('open', 0)} 筆／盤中當下買(intraday) {exec_cnt.get('intraday', 0)} 筆／收盤買進(close) {exec_cnt.get('close', 0)} 筆")
        print(f"勝率 {m['win']:.1f}%｜平均 {m['avg_ret']:.2f}%｜中位數 {m['med_ret']:.2f}%｜"
              f"獲利因子 {m['pf']:.2f}｜平均持有 {m['avg_hold']:.1f} 日")
        print(f"Calmar {fmt(rk['calmar'])}｜Martin {fmt(rk['martin'])}（無風險 {a.rf:g}%、Ulcer {fmt(rk['ulcer'])}）｜"
              f"恢復因子 {fmt(rk['recovery'])}（淨利 {m['final'] - bt['Initial_Capital']:,.0f} ÷ 最大回撤金額 {fmt(rk['mdd_amt'], 0)}）")
        if "side" in trades.columns and (trades["side"] == "SHORT").any():
            sh_df = trades[trades["side"] == "SHORT"]
            lg_df = trades[trades["side"] == "LONG"]
            print(f"多空拆解：多單 {len(lg_df)} 筆（勝率 {(lg_df['pnl']>0).mean()*100:.1f}% / 平均 {lg_df['return_pct'].mean():+.2f}%）｜"
                  f"跳空放空 {len(sh_df)} 筆（勝率 {(sh_df['pnl']>0).mean()*100:.1f}% / 平均 {sh_df['return_pct'].mean():+.2f}%）")
            print(f"跳空放空明細：總觸發 {st['gap_short_cnt']} 筆 → 盤中停損 {st['gap_short_sl']} 筆 / "
                  f"鎖漲停隔日補 {st['gap_short_limit_next']} 筆 / 收盤正常回補並反手作多 {st['gap_short_reverse_long']} 筆")
        print(f"過濾統計：通過規模/均量 {st['eligible']} 檔次 → 因買進時點漲跌幅未達標跳過 {st['skip_buy_chg']} 檔次／防追高過濾跳過 {st.get('skip_chase', 0)} 檔次")
        if "side" in trades.columns and (trades["side"] == "ETF").any():
            stk_df = trades[trades["side"] != "ETF"]
            etf_df = trades[trades["side"] == "ETF"]
            print(f"資產收益拆解：個股動能 {len(stk_df)} 筆（淨損益 {stk_df['pnl'].sum():+,.0f} 元 / 勝率 {(stk_df['pnl']>0).mean()*100:.1f}%）｜"
                  f"{p.get('Idle_ETF_Code', '0050')} 閒置停泊（已實現損益 {etf_df['pnl'].sum():+,.0f} 元 / 買入 {st['etf_buy_cnt']} 次 / 賣出 {st['etf_sell_cnt']} 次 / 獨立停損 {st['etf_sl_cnt']} 次）")
        print(f"出場方式：{ex}｜期末未平倉 {m['unclosed']} 檔｜對照基準 {base:.2f}%/筆（策略超額 {m['avg_ret'] - base:+.2f}%）")
    else:
        print("⚠️ 期間內無交易")
    print("=" * 88)

    png = os.path.join(run_dir, f"{tag_of(p)}.png")
    draw_chart(png, p, bt, start, end, eq, trades, m, rk, a.rf, st)
    if len(trades):
        trades.to_csv(os.path.join(run_dir, "trades.csv"), index=False, encoding="utf-8-sig")
    eq.to_csv(os.path.join(run_dir, "equity.csv"), encoding="utf-8-sig")
    print(f"💾 圖表與明細已輸出：{run_dir}")


if __name__ == "__main__":
    main()
