from enum import StrEnum


class EntryType(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    INSIGHT = "INSIGHT"
    MARKET_EVENT = "MARKET_EVENT"
    REFLECTION = "REFLECTION"


class Horizon(StrEnum):
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


class Emotion(StrEnum):
    CONFIDENT = "confident"
    EXCITED = "excited"
    CURIOUS = "curious"
    CALM = "calm"
    UNCERTAIN = "uncertain"
    ANXIOUS = "anxious"
    FOMO = "fomo"
    REGRET = "regret"
    RELIEVED = "relieved"
    FRUSTRATED = "frustrated"


class IdeaSource(StrEnum):
    OWN_RESEARCH = "own_research"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    FRIEND_OR_FAMILY = "friend_or_family"
    PROFESSIONAL = "professional"


# what the writer's own reasoning for the decision rests on
class ReasoningBasis(StrEnum):
    FUNDAMENTALS = "fundamentals"
    PRICE_ACTION = "price_action"
    COMMENTARY = "commentary"
    MIXED = "mixed"


# set only when an entry looks back on whether an earlier thesis held
class ThesisOutcome(StrEnum):
    PLAYED_OUT = "played_out"
    DID_NOT = "did_not"
    MIXED = "mixed"


# vestry skips kinds it doesn't know, so adding one is backward compatible
class PatternKind(StrEnum):
    FELT_VS_DID = "felt_vs_did"
    OUTCOMES_BY_STATE = "outcomes_by_state"
    HORIZON_VS_HOLDING = "horizon_vs_holding"
    GAINS_VS_LOSSES = "gains_vs_losses"
    SILENT_TRADES = "silent_trades"
    OPEN_THESES = "open_theses"


# vestry skips kinds it doesn't know, so adding one is backward compatible
class BehaviorKind(StrEnum):
    NO_PLAN = "no_plan"
    UNDOCUMENTED = "undocumented"
    EMOTIONAL = "emotional"
    FOMO = "fomo"
    HYPE = "hype"
    RUN_UP = "run_up"
    OVERTRADING = "overtrading"
    OVERCONFIDENCE = "overconfidence"
    PRICE_ACTION = "price_action"
    LOSS_AVERSION = "loss_aversion"
    SHORT_TERM = "short_term"
    RISK_CONCENTRATION = "risk_concentration"
    POSITION_SIZING = "position_sizing"
