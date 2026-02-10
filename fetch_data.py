#!/usr/bin/env python3
"""
Global Peer Valuation Monitor - Data Fetcher
yfinance 밸류에이션 + structural-shortage 데이터 통합
"""

import json
import time
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yfinance as yf

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
OUTPUT_PATH = BASE_DIR / "data" / "valuation-data.json"
HISTORY_DIR = BASE_DIR / "data" / "history"
HISTORY_INDEX = HISTORY_DIR / "index.json"

KST = timezone(timedelta(hours=9))


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_shortage_data(config):
    """쇼티지 대시보드 데이터를 로드하여 아이템명 → 데이터 맵 반환."""
    shortage_path = BASE_DIR / config.get("shortagePath", "")
    if not shortage_path.exists():
        print(f"  Shortage data not found: {shortage_path}")
        return {}

    with open(shortage_path) as f:
        data = json.load(f)

    item_map = {}
    for item in data.get("items", []):
        item_map[item["name"]] = {
            "alertLevel": item.get("alertLevel"),
            "priceYoY": item.get("priceYoY"),
            "utilization": item.get("utilization"),
            "inventory": item.get("inventory"),
            "leadTime": item.get("leadTime"),
            "category": item.get("category"),
            "priceData": item.get("priceData"),
        }
    return item_map


def fetch_stock_data(symbol: str) -> dict:
    """단일 종목의 밸류에이션 데이터를 수집."""
    ticker = yf.Ticker(symbol)
    info = ticker.info

    result = {
        "symbol": symbol,
        "currentPrice": info.get("currentPrice"),
        "currency": info.get("currency"),
        "marketCap": info.get("marketCap"),
        "forwardPE": info.get("forwardPE"),
        "trailingPE": info.get("trailingPE"),
        "priceToBook": info.get("priceToBook"),
        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        "dividendYield": info.get("dividendYield"),
        "returnOnEquity": info.get("returnOnEquity"),
        "forwardEps": info.get("forwardEps"),
        "trailingEps": info.get("trailingEps"),
        "targetMeanPrice": info.get("targetMeanPrice"),
        "targetMedianPrice": info.get("targetMedianPrice"),
        "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
        "recommendationMean": info.get("recommendationMean"),
    }

    # earnings_estimate에서 forward consensus
    try:
        ee = ticker.earnings_estimate
        if ee is not None and not ee.empty:
            for period in ["0y", "+1y"]:
                if period in ee.index:
                    row = ee.loc[period]
                    prefix = "fwd0y" if period == "0y" else "fwd1y"
                    result[f"{prefix}_epsAvg"] = _safe_float(row.get("avg"))
                    result[f"{prefix}_epsLow"] = _safe_float(row.get("low"))
                    result[f"{prefix}_epsHigh"] = _safe_float(row.get("high"))
                    result[f"{prefix}_numAnalysts"] = _safe_int(row.get("numberOfAnalysts"))
                    result[f"{prefix}_growth"] = _safe_float(row.get("growth"))
    except Exception:
        pass

    # EPS revision (30일 전 대비)
    try:
        et = ticker.eps_trend
        if et is not None and not et.empty and "+1y" in et.index:
            row = et.loc["+1y"]
            current = _safe_float(row.get("current"))
            ago_30d = _safe_float(row.get("30daysAgo"))
            if current is not None and ago_30d is not None and ago_30d != 0:
                result["epsRevision30d"] = round((current - ago_30d) / abs(ago_30d) * 100, 2)
    except Exception:
        pass

    # target upside
    price = result.get("currentPrice")
    target = result.get("targetMeanPrice")
    if price and target and price > 0:
        result["targetUpside"] = round((target - price) / price * 100, 2)

    # forward P/E 보정
    if result.get("forwardPE") is None and price and result.get("fwd1y_epsAvg"):
        eps = result["fwd1y_epsAvg"]
        if eps > 0:
            result["forwardPE"] = round(price / eps, 2)

    return result


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:
            return None
        return round(f, 4)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def format_market_cap(mc):
    if mc is None:
        return None
    if mc >= 1e12:
        return f"${mc/1e12:.1f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.1f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.0f}M"
    return str(mc)


def _calc_peer_avg(stocks: list) -> dict:
    GROWTH_CAP = 10  # ±1000% — exclude extreme base-effect outliers
    metrics = [
        "forwardPE", "trailingPE", "priceToBook", "enterpriseToEbitda",
        "dividendYield", "returnOnEquity", "fwd1y_growth", "targetUpside",
        "epsRevision30d",
    ]
    avg = {}
    for m in metrics:
        values = [s[m] for s in stocks if s.get(m) is not None]
        if m == "fwd1y_growth":
            values = [v for v in values if abs(v) <= GROWTH_CAP]
        if values:
            avg[m] = round(sum(values) / len(values), 4)
            values_sorted = sorted(values)
            mid = len(values_sorted) // 2
            if len(values_sorted) % 2 == 0 and len(values_sorted) >= 2:
                avg[f"{m}_median"] = round((values_sorted[mid - 1] + values_sorted[mid]) / 2, 4)
            else:
                avg[f"{m}_median"] = values_sorted[mid]
    return avg


_ticker_cache = {}


def fetch_stock_data_cached(symbol: str) -> dict:
    """캐시된 티커 데이터 반환. 같은 티커 중복 호출 방지."""
    if symbol in _ticker_cache:
        return _ticker_cache[symbol].copy()
    data = fetch_stock_data(symbol)
    _ticker_cache[symbol] = data
    return data.copy()


def main():
    config = load_config()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 쇼티지 데이터 로드
    print("Loading shortage data...")
    shortage_map = load_shortage_data(config)
    print(f"  {len(shortage_map)} shortage items loaded")

    total_stocks = sum(len(c["stocks"]) for c in config["categories"])
    print(f"\nFetching {total_stocks} stocks across {len(config['categories'])} categories...")

    output = {
        "fetchedAt": datetime.now(KST).isoformat(),
        "categories": [],
    }

    count = 0
    errors = []

    for cat in config["categories"]:
        cat_data = {
            "id": cat["id"],
            "name": cat["name"],
            "name_en": cat["name_en"],
            "group": cat.get("group", ""),
            "stocks": [],
            "shortage": [],
        }

        # 쇼티지 아이템 매칭
        for item_name in cat.get("shortageItems", []):
            if item_name in shortage_map:
                s = shortage_map[item_name]
                cat_data["shortage"].append({
                    "name": item_name,
                    "alertLevel": s["alertLevel"],
                    "priceYoY": s.get("priceYoY"),
                    "utilization": s.get("utilization"),
                    "inventory": s.get("inventory"),
                    "leadTime": s.get("leadTime"),
                })

        # 카테고리 전체 쇼티지 요약
        alerts = [s["alertLevel"] for s in cat_data["shortage"]]
        if alerts:
            if "red" in alerts:
                cat_data["alertSummary"] = "red"
            elif "yellow" in alerts:
                cat_data["alertSummary"] = "yellow"
            else:
                cat_data["alertSummary"] = "green"
            cat_data["shortageCount"] = len(alerts)

        # 주식 데이터 수집
        for stock_cfg in cat["stocks"]:
            count += 1
            symbol = stock_cfg["symbol"]
            name = stock_cfg["name"]
            was_cached = symbol in _ticker_cache
            print(f"  [{count}/{total_stocks}] {name} ({symbol})...", end=" ", flush=True)

            try:
                data = fetch_stock_data_cached(symbol)
                data["name"] = name
                data["country"] = stock_cfg["country"]
                data["marketCapFormatted"] = format_market_cap(data.get("marketCap"))
                cat_data["stocks"].append(data)
                print("OK (cached)" if was_cached else "OK")
            except Exception as e:
                print(f"ERROR: {e}")
                errors.append({"symbol": symbol, "name": name, "error": str(e)})
                cat_data["stocks"].append({
                    "symbol": symbol,
                    "name": name,
                    "country": stock_cfg["country"],
                    "error": str(e),
                })

            if not was_cached:
                time.sleep(1.5)

        # 피어 평균
        valid = [s for s in cat_data["stocks"] if "error" not in s]
        if valid:
            cat_data["peerAvg"] = _calc_peer_avg(valid)

        output["categories"].append(cat_data)

    output["errors"] = errors
    output["totalStocks"] = total_stocks
    output["successCount"] = total_stocks - len(errors)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {output['successCount']}/{total_stocks} stocks fetched.")
    print(f"Errors: {len(errors)}")
    print(f"Output: {OUTPUT_PATH}")

    # 히스토리 스냅샷 저장 + 변동률 계산
    save_snapshot(output)
    calculate_changes(output)


def save_snapshot(output):
    """당일 compact 스냅샷을 data/history/에 저장."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(KST).strftime("%Y-%m-%d")

    snapshot = {"date": today, "fetchedAt": output["fetchedAt"], "stocks": []}

    for cat in output["categories"]:
        for stock in cat["stocks"]:
            if "error" in stock:
                continue
            snapshot["stocks"].append({
                "s": stock["symbol"],
                "c": cat["id"],
                "p": stock.get("currentPrice"),
                "fpe": stock.get("forwardPE"),
                "tpe": stock.get("trailingPE"),
                "pb": stock.get("priceToBook"),
                "ev": stock.get("enterpriseToEbitda"),
                "tu": stock.get("targetUpside"),
                "eg": stock.get("fwd1y_growth"),
                "er": stock.get("epsRevision30d"),
            })

    snap_path = HISTORY_DIR / f"{today}.json"
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, separators=(",", ":"))

    # index.json 갱신
    if HISTORY_INDEX.exists():
        with open(HISTORY_INDEX) as f:
            index = json.load(f)
    else:
        index = {"dates": []}

    if today not in index["dates"]:
        index["dates"].append(today)
        index["dates"].sort()

    index["firstDate"] = index["dates"][0]
    index["lastDate"] = index["dates"][-1]
    index["count"] = len(index["dates"])

    with open(HISTORY_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nSnapshot saved: {snap_path} ({snap_path.stat().st_size / 1024:.1f}KB)")


def calculate_changes(output):
    """히스토리 스냅샷에서 Δ1D/7D/30D Forward P/E 변동률을 계산하여 output에 추가."""
    today = datetime.now(KST).date()

    def find_snapshot(target_date, max_gap=3):
        for offset in range(1, max_gap + 1):
            for delta in [-offset, offset]:
                check = target_date + timedelta(days=delta)
                path = HISTORY_DIR / f"{check.isoformat()}.json"
                if path.exists():
                    with open(path) as f:
                        return json.load(f)
        return None

    periods = {
        "change1d": find_snapshot(today, max_gap=3),
        "change7d": find_snapshot(today - timedelta(days=6), max_gap=3),
        "change30d": find_snapshot(today - timedelta(days=29), max_gap=5),
    }

    # symbol → fpe 맵 생성
    history_maps = {}
    for key, snap in periods.items():
        if snap:
            history_maps[key] = {s["s"]: s.get("fpe") for s in snap["stocks"]}
        else:
            history_maps[key] = {}

    changes_found = 0
    for cat in output["categories"]:
        for stock in cat["stocks"]:
            if "error" in stock:
                continue
            current_pe = stock.get("forwardPE")
            for key, hmap in history_maps.items():
                old_pe = hmap.get(stock["symbol"])
                if current_pe is not None and old_pe is not None and old_pe != 0:
                    stock[key] = round((current_pe - old_pe) / old_pe * 100, 1)
                    changes_found += 1
                else:
                    stock[key] = None

    # 변동률 포함하여 다시 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Changes calculated: {changes_found} values across {len(history_maps)} periods")


if __name__ == "__main__":
    main()
