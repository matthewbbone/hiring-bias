from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

__all__ = [
    "ChoiceDist",
    "TransitionDist",
    "OccupationDist",
    "SalaryDist",
    "VacancyDist",
    "choice_dist_from_dict",
    "choice_dist_to_dict",
    "occupation_dist_from_dict",
    "occupation_dist_to_dict",
    "vacancy_dist_from_dict",
    "vacancy_dist_to_dict",
]


@dataclass
class ChoiceDist:
    choices: list[str]
    freq: list[float]

    def __getitem__(self, key: str) -> list[str] | list[float]:
        if key == "CHOICES":
            return self.choices
        if key == "FREQ":
            return self.freq
        raise KeyError(key)

    def __setitem__(self, key: str, value: list[str] | list[float]) -> None:
        if key == "CHOICES":
            self.choices = value  # type: ignore[assignment]
            return
        if key == "FREQ":
            self.freq = value  # type: ignore[assignment]
            return
        raise KeyError(key)

    def to_dict(self) -> dict[str, list[str] | list[float]]:
        return {
            "CHOICES": self.choices,
            "FREQ": self.freq,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | "ChoiceDist") -> "ChoiceDist":
        if isinstance(data, ChoiceDist):
            return data
        return cls(
            choices=list(data["CHOICES"]),  # type: ignore[index]
            freq=list(data["FREQ"]),  # type: ignore[index]
        )


@dataclass
class TransitionDist(ChoiceDist):
    prob: list[float] | None = None
    total: Optional[float] = None

    def __getitem__(self, key: str) -> list[str] | list[float] | Optional[float]:
        if key == "PROB":
            return self.prob
        if key == "TOTAL":
            return self.total
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: list[str] | list[float] | Optional[float]) -> None:
        if key == "PROB":
            self.prob = value  # type: ignore[assignment]
            return
        if key == "TOTAL":
            self.total = value  # type: ignore[assignment]
            return
        super().__setitem__(key, value)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "CHOICES": self.choices,
            "FREQ": self.freq,
        }
        if self.prob is not None:
            data["PROB"] = self.prob
        if self.total is not None:
            data["TOTAL"] = self.total
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | "TransitionDist") -> "TransitionDist":
        if isinstance(data, TransitionDist):
            return data
        return cls(
            choices=list(data["CHOICES"]),  # type: ignore[index]
            freq=list(data["FREQ"]),  # type: ignore[index]
            prob=list(data["PROB"]) if "PROB" in data else None,  # type: ignore[index]
            total=data.get("TOTAL"),  # type: ignore[arg-type]
        )


_OCC_KEY_TO_ATTR: dict[str, str] = {
    "DEGREE_MAJORS": "degree_majors",
    "SKILLS": "skills",
    "TRANSITION_FROM": "transition_from",
    "TRANSITION_TO": "transition_to",
    "EXPERIENCES": "experiences",
}


@dataclass
class OccupationDist:
    degree_majors: Optional[ChoiceDist] = None
    skills: Optional[ChoiceDist] = None
    transition_from: Optional[TransitionDist] = None
    transition_to: Optional[TransitionDist] = None
    experiences: Optional[ChoiceDist] = None
    soc_name: str = ""
    count: int = 0

    def __getitem__(self, key: str) -> ChoiceDist | TransitionDist:
        attr = _OCC_KEY_TO_ATTR.get(key)
        if attr is None:
            raise KeyError(key)
        value = getattr(self, attr)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        attr = _OCC_KEY_TO_ATTR.get(key)
        if attr is None:
            return False
        return getattr(self, attr) is not None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {}
        if self.degree_majors is not None:
            data["DEGREE_MAJORS"] = self.degree_majors.to_dict()
        if self.skills is not None:
            data["SKILLS"] = self.skills.to_dict()
        if self.transition_from is not None:
            data["TRANSITION_FROM"] = self.transition_from.to_dict()
        if self.transition_to is not None:
            data["TRANSITION_TO"] = self.transition_to.to_dict()
        if self.experiences is not None:
            data["EXPERIENCES"] = self.experiences.to_dict()
        data["SOC_NAME"] = self.soc_name
        data["COUNT"] = self.count
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | "OccupationDist") -> "OccupationDist":
        if isinstance(data, OccupationDist):
            return data
        return cls(
            degree_majors=choice_dist_from_dict(data["DEGREE_MAJORS"]) if "DEGREE_MAJORS" in data else None,  # type: ignore[arg-type]
            skills=choice_dist_from_dict(data["SKILLS"]) if "SKILLS" in data else None,  # type: ignore[arg-type]
            transition_from=TransitionDist.from_dict(data["TRANSITION_FROM"]) if "TRANSITION_FROM" in data else None,  # type: ignore[arg-type]
            transition_to=TransitionDist.from_dict(data["TRANSITION_TO"]) if "TRANSITION_TO" in data else None,  # type: ignore[arg-type]
            experiences=choice_dist_from_dict(data["EXPERIENCES"]) if "EXPERIENCES" in data else None,  # type: ignore[arg-type]
            soc_name=str(data.get("SOC_NAME", "")),
            count=int(data.get("COUNT", 0)),  # type: ignore[union-attr]
        )


@dataclass
class SalaryDist:
    mu: float
    sigma: float

    def __getitem__(self, key: str) -> float:
        if key == "MU":
            return self.mu
        if key == "SIGMA":
            return self.sigma
        raise KeyError(key)

    def to_dict(self) -> dict[str, float]:
        return {
            "MU": self.mu,
            "SIGMA": self.sigma,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | "SalaryDist") -> "SalaryDist":
        if isinstance(data, SalaryDist):
            return data
        return cls(
            mu=float(data["MU"]),  # type: ignore[index]
            sigma=float(data["SIGMA"]),  # type: ignore[index]
        )


_VAC_KEY_TO_ATTR: dict[str, str] = {
    "COMPANY_NAME": "company_name",
    "TITLE": "title",
    "SKILLS": "skills",
    "SALARY": "salary",
    "SOC_NAME": "soc_name",
}


@dataclass
class VacancyDist:
    company_name: ChoiceDist
    title: ChoiceDist
    skills: ChoiceDist
    salary: SalaryDist
    soc_name: str
    count: int = 0

    def __getitem__(self, key: str) -> ChoiceDist | SalaryDist | str:
        attr = _VAC_KEY_TO_ATTR.get(key)
        if attr is None:
            raise KeyError(key)
        return getattr(self, attr)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in _VAC_KEY_TO_ATTR

    def to_dict(self) -> dict[str, object]:
        return {
            "COMPANY_NAME": self.company_name.to_dict(),
            "TITLE": self.title.to_dict(),
            "SKILLS": self.skills.to_dict(),
            "SALARY": self.salary.to_dict(),
            "SOC_NAME": self.soc_name,
            "COUNT": self.count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | "VacancyDist") -> "VacancyDist":
        if isinstance(data, VacancyDist):
            return data
        return cls(
            company_name=choice_dist_from_dict(data["COMPANY_NAME"]),  # type: ignore[arg-type]
            title=choice_dist_from_dict(data["TITLE"]),  # type: ignore[arg-type]
            skills=choice_dist_from_dict(data["SKILLS"]),  # type: ignore[arg-type]
            salary=SalaryDist.from_dict(data["SALARY"]),  # type: ignore[arg-type]
            soc_name=str(data.get("SOC_NAME", "")),
            count=int(data.get("COUNT", 0)),  # type: ignore[arg-type]
        )


def choice_dist_from_dict(data: Mapping[str, object] | ChoiceDist) -> ChoiceDist:
    return ChoiceDist.from_dict(data)


def choice_dist_to_dict(dist: ChoiceDist) -> dict[str, list[str] | list[float]]:
    return dist.to_dict()


def occupation_dist_from_dict(data: Mapping[str, object] | OccupationDist) -> OccupationDist:
    return OccupationDist.from_dict(data)


def occupation_dist_to_dict(dist: OccupationDist) -> dict[str, object]:
    return dist.to_dict()


def vacancy_dist_from_dict(data: Mapping[str, object] | VacancyDist) -> VacancyDist:
    return VacancyDist.from_dict(data)

def vacancy_dist_to_dict(dist: VacancyDist) -> dict[str, object]:
    return dist.to_dict()
