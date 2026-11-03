from dataclasses import dataclass
from typing import Optional

@dataclass
class Vacancy:

    company_name: str
    title: str
    soc_code: str
    soc_name: str
    education_level: str
    skills: list[str]
    salary: int
    perturbed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.company_name, str):
            raise TypeError("company_name must be str")
        if not isinstance(self.title, str):
            raise TypeError("title must be str")
        if not isinstance(self.soc_code, str):
            raise TypeError("soc_code must be str")
        if not isinstance(self.soc_name, str):
            raise TypeError("soc_name must be str")
        if not isinstance(self.education_level, str):
            raise TypeError("education_level must be str")
        if not isinstance(self.skills, list):
            raise TypeError("skills must be list")
        if not isinstance(self.salary, int):
            raise TypeError("salary must be int")
        if not isinstance(self.perturbed, bool):
            raise TypeError("perturbed must be bool")


@dataclass
class Profile:

    age: int
    first_name: str
    last_name: str
    soc_code: str
    soc_name: str
    degree_major: str
    school: str
    skills: list[str]
    experience: str
    grad_year: int
    
    perturbed: bool = False
    
    # Set during perturbation
    race: Optional[int] = None
    gender: Optional[int] = None
    over40: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.age, int):
            raise TypeError("age must be int")
        if not isinstance(self.first_name, str):
            raise TypeError("first_name must be str")
        if not isinstance(self.last_name, str):
            raise TypeError("last_name must be str")
        if not isinstance(self.soc_code, str):
            raise TypeError("soc_code must be str")
        if not isinstance(self.soc_name, str):
            raise TypeError("soc_name must be str")
        if self.gender is not None and not isinstance(self.gender, int):
            raise TypeError("gender must be int or None")
        if not isinstance(self.degree_major, str):
            raise TypeError("degree_major must be str")
        if not isinstance(self.school, str):
            raise TypeError("school must be str")
        if not isinstance(self.skills, list):
            raise TypeError("skills must be list")
        if not isinstance(self.experience, str):
            raise TypeError("experience must be str")
        if self.race is not None and not isinstance(self.race, int):
            raise TypeError("race must be int or None")
        if self.over40 is not None and not isinstance(self.over40, int):
            raise TypeError("over40 must be int or None")
        if not isinstance(self.perturbed, bool):
            raise TypeError("perturbed must be bool")
