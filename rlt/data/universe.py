"""Instrument specifications and research universes.

Cost parameters are deliberately explicit, because "costs" in RL trading papers are often a single
flat number. Per side, a trade of notional X in instrument i pays

    commission   X * commission_bps            (equities/ETFs)   or  |contracts| * commission_per_contract (futures)
    half-spread  X * max(min_half_spread_bps, 0.5 * tick / price)        tick = 1/16 $ before decimalisation
    slippage     X * slippage_bps (environment setting)
    impact       X * impact_coef * sigma_daily * sqrt(X / ADV$)          square-root law (Almgren et al. 2005)

Short equity/ETF positions pay `borrow_bps_annual`; cash earns (or pays) the T-bill rate. Futures post no
notional (only margin) and pay no borrow. All of this is applied in rlt.env.

Universe choices
----------------
* `spy`        single asset, 1993-2026: the starting point of the study.
* `etf9`       cross-asset ETF set with a common history from Nov-2004 (GLD inception).
* `eq10_2000`  the ten largest S&P 500 members by market value at 31-Dec-2000 (GE, XOM, PFE, C, CSCO, WMT,
               MSFT, AIG, MRK, INTC). Picked by what was big *then*, not by what did well since, to avoid the
               survivorship/hindsight bias of today's mega-caps. It contains famous losers (C, AIG, GE, INTC, CSCO).
* `fut4`       ES, NQ, YM e-minis (roll artefacts in Yahoo's continuous series rebuilt from the cash index) and ZN
               10y note (contract-switch days rebuilt from IEF with a trailing regression beta; the raw series
               loses about 60 bp on each quarterly switch, roughly 2-2.5%/yr of fictitious negative drift).
* `eq10_2000` and `pool19` train only from 2001-01-02 onwards: the stocks were selected by their size at
               31-Dec-2000, so 1994-2000 returns would carry the selection's hindsight (+29%/yr vs SPY +17%).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    asset_class: str  # "equity" | "etf" | "future"
    source: str  # Yahoo symbol
    multiplier: float = 1.0
    tick_size: float = 0.01
    commission_bps: float = 0.3
    commission_per_contract: float = 0.0
    min_half_spread_bps: float = 0.5
    impact_coef: float = 0.5
    borrow_bps_annual: float = 30.0
    margin_rate: float = 0.0  # futures initial margin as a fraction of notional (informational + leverage check)
    shortable: bool = True
    spot_proxy: str | None = None  # cash index (or related ETF) used to rebuild futures roll days
    roll_months: tuple = field(default_factory=tuple)
    proxy_beta: str = "one"  # "one": proxy is the futures' own index; "rolling": trailing regression beta

    @property
    def is_future(self) -> bool:
        return self.asset_class == "future"


def _etf(sym, hs=0.5, borrow=30.0):
    return InstrumentSpec(sym, "etf", sym, commission_bps=0.3, min_half_spread_bps=hs, impact_coef=0.5,
                          borrow_bps_annual=borrow)


def _eq(sym, hs=1.0):
    return InstrumentSpec(sym, "equity", sym, commission_bps=0.5, min_half_spread_bps=hs, impact_coef=0.7,
                          borrow_bps_annual=40.0)


_QTR = (3, 6, 9, 12)

SPECS: dict[str, InstrumentSpec] = {
    # ETFs
    "SPY": _etf("SPY", 0.3, 25.0),
    "QQQ": _etf("QQQ", 0.4, 25.0),
    "IWM": _etf("IWM", 0.5, 30.0),
    "EFA": _etf("EFA", 0.7, 30.0),
    "EEM": _etf("EEM", 1.0, 40.0),
    "TLT": _etf("TLT", 0.5, 30.0),
    "IEF": _etf("IEF", 0.5, 30.0),
    "LQD": _etf("LQD", 0.8, 40.0),
    "GLD": _etf("GLD", 0.5, 30.0),
    # single stocks: largest S&P 500 members at end-2000
    **{s: _eq(s) for s in ["GE", "XOM", "PFE", "C", "CSCO", "WMT", "MSFT", "AIG", "MRK", "INTC"]},
    # futures
    "ES": InstrumentSpec("ES", "future", "ES=F", multiplier=50.0, tick_size=0.25, commission_per_contract=2.25,
                         min_half_spread_bps=0.2, impact_coef=0.4, margin_rate=0.06, spot_proxy="^GSPC",
                         roll_months=_QTR),
    "NQ": InstrumentSpec("NQ", "future", "NQ=F", multiplier=20.0, tick_size=0.25, commission_per_contract=2.25,
                         min_half_spread_bps=0.3, impact_coef=0.4, margin_rate=0.07, spot_proxy="^NDX",
                         roll_months=_QTR),
    "YM": InstrumentSpec("YM", "future", "YM=F", multiplier=5.0, tick_size=1.0, commission_per_contract=2.25,
                         min_half_spread_bps=0.4, impact_coef=0.5, margin_rate=0.06, spot_proxy="^DJI",
                         roll_months=_QTR),
    "ZN": InstrumentSpec("ZN", "future", "ZN=F", multiplier=1000.0, tick_size=1 / 64, commission_per_contract=2.0,
                         min_half_spread_bps=0.3, impact_coef=0.3, margin_rate=0.025, spot_proxy="IEF",
                         roll_months=_QTR, proxy_beta="rolling"),
}

UNIVERSES: dict[str, list[str]] = {
    "spy": ["SPY"],
    "etf9": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD"],
    "eq10_2000": ["GE", "XOM", "PFE", "C", "CSCO", "WMT", "MSFT", "AIG", "MRK", "INTC"],
    "fut4": ["ES", "NQ", "YM", "ZN"],
}
# pooled single-instrument training set: SPY + the other ETFs + the end-2000 large caps
UNIVERSES["pool19"] = ["SPY"] + [s for s in UNIVERSES["etf9"] if s != "SPY"] + UNIVERSES["eq10_2000"]


# SEC emergency order (19 Sep - 8 Oct 2008) banned new short sales in ~800 financial stocks; of this lab's
# instruments that covers Citigroup and AIG. Existing shorts could be covered.
SHORT_BANS: dict[str, list] = {"C": [("2008-09-19", "2008-10-08")], "AIG": [("2008-09-19", "2008-10-08")]}


def get_spec(symbol: str) -> InstrumentSpec:
    if symbol in SPECS:
        return SPECS[symbol]
    # unknown tickers default to a liquid-equity cost profile
    return _eq(symbol)
