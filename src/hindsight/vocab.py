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
