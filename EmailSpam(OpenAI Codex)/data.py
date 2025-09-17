"""Synthetic email dataset for the spam consistency example."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class EmailExample:
    header: str
    body: str
    label: int  # 1 = spam, 0 = legitimate

    def to_record(self) -> Dict[str, object]:
        return {
            "header": self.header,
            "body": self.body,
            "label": self.label,
        }


def build_dataset() -> Dict[str, List[Dict[str, object]]]:
    """Return train/dev/test splits with simple spam heuristics."""
    raw: Sequence[EmailExample] = (
        EmailExample("Limited time offer", "Save 70% today only. Act now!", 1),
        EmailExample("Weekly report", "Attached is this week's performance summary.", 0),
        EmailExample("Claim your prize", "You've been selected for a luxury cruise.", 1),
        EmailExample("Lunch tomorrow?", "Can we meet at noon for lunch at the cafe?", 0),
        EmailExample("Re: Budget discussion", "Let's revisit the Q3 budget adjustments.", 0),
        EmailExample("Win a free phone", "Click to win the latest smartphone giveaway!", 1),
        EmailExample("Invoice available", "Your invoice for services rendered is attached.", 0),
        EmailExample("Investment alert", "Guaranteed returns! Double your money fast.", 1),
        EmailExample("Staff meeting notes", "Here are the minutes from today's meeting.", 0),
        EmailExample("Urgent: Verify your account", "Security alert! Reset your password immediately.", 1),
        EmailExample("Family reunion", "Looking forward to seeing everyone this weekend.", 0),
        EmailExample("Bonus unlocked", "Congratulations, your bonus is waiting—confirm now.", 1),
    )

    records = [item.to_record() for item in raw]

    return {
        "train": records[:8],
        "dev": records[8:10],
        "test": records[10:],
    }
