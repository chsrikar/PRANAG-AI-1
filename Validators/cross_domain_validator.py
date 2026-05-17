"""
cross_domain_validator.py  —  DIVYANSHU  (Task 2)
Sequential: Biology → Materials → Physics → Chemistry
If any domain fails → STOP and return failure reason.
"""

from dataclasses import dataclass, field
from typing import Optional
from validator import SimulationResult


DOMAIN_THRESHOLDS = {
    "biology":   0.65,
    "materials": 0.68,
    "physics":   0.70,
    "chemistry": 0.72,
}
DOMAIN_ORDER = ["biology", "materials", "physics", "chemistry"]


@dataclass
class DomainCheck:
    domain:    str
    score:     float
    threshold: float
    passed:    bool
    reason:    Optional[str] = None


@dataclass
class CrossDomainResult:
    design_id:      str
    overall_passed: bool
    checks:         list = field(default_factory=list)
    failure_domain: Optional[str] = None
    failure_reason: Optional[str] = None
    domains_checked:int = 0

    def to_dict(self):
        return {
            "design_id":      self.design_id,
            "overall_passed": self.overall_passed,
            "failure_domain": self.failure_domain,
            "failure_reason": self.failure_reason,
            "domains_checked":self.domains_checked,
            "checks": [{
                "domain":    c.domain,
                "score":     round(c.score, 4),
                "threshold": c.threshold,
                "passed":    c.passed,
                "reason":    c.reason,
            } for c in self.checks],
        }


class CrossDomainValidator:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or DOMAIN_THRESHOLDS

    def _check(self, sim: SimulationResult, domain: str) -> DomainCheck:
        score  = getattr(sim, f"{domain}_score", 0.0)
        thresh = self.thresholds[domain]
        passed = score >= thresh
        return DomainCheck(
            domain    = domain,
            score     = score,
            threshold = thresh,
            passed    = passed,
            reason    = None if passed else
                        f"{domain.capitalize()} score {score:.4f} < min {thresh}",
        )

    def validate(self, sim: SimulationResult) -> CrossDomainResult:
        result = CrossDomainResult(design_id=sim.design_id, overall_passed=False)
        for domain in DOMAIN_ORDER:
            check = self._check(sim, domain)
            result.checks.append(check)
            result.domains_checked += 1
            if not check.passed:
                result.failure_domain = domain
                result.failure_reason = check.reason
                return result   # ← STOP immediately
        result.overall_passed = True
        return result

    def validate_batch(self, simulations: list) -> dict:
        results  = [self.validate(s) for s in simulations]
        passed   = [r for r in results if r.overall_passed]
        failed   = [r for r in results if not r.overall_passed]
        fail_dist= {d: 0 for d in DOMAIN_ORDER}
        for r in failed:
            if r.failure_domain:
                fail_dist[r.failure_domain] += 1

        return {
            "total":                len(results),
            "passed":               len(passed),
            "failed":               len(failed),
            "failure_distribution": fail_dist,
            "results":              [r.to_dict() for r in results],
        }


if __name__ == "__main__":
    from validator import load_mock_simulations
    sims   = load_mock_simulations(40)
    cdv    = CrossDomainValidator()
    report = cdv.validate_batch(sims)
    print(f"Cross-Domain → Passed: {report['passed']}/{report['total']}")
    print(f"Failure dist: {report['failure_distribution']}")
