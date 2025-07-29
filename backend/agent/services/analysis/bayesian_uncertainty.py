"""Bayesian uncertainty handling system for fact verification.

from __future__ import annotations

This module implements sophisticated uncertainty quantification using Bayesian methods,
providing confidence intervals, uncertainty propagation, and probabilistic reasoning.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from app.exceptions import AnalysisError

warnings.filterwarnings("ignore", category=FutureWarning)

# Lazy imports for heavy libraries
pm = None
az = None


def _ensure_bayesian_libraries():
    """Ensure PyMC and ArviZ are imported (lazy initialization)."""
    global pm, az
    if pm is None or az is None:
        logger.info(
            "Initializing Bayesian libraries (PyMC and ArviZ) - lazy initialization")
        try:
            import arviz as az_module
            import pymc as pm_module

            pm = pm_module
            az = az_module
            logger.info("Bayesian libraries initialized successfully")
        except ImportError as e:
            raise AnalysisError(
                f"Bayesian libraries not available: {e}") from e


logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for Bayesian uncertainty handling."""

    use_bayesian_inference: bool = True
    mcmc_samples: int = 2000
    mcmc_tune: int = 1000
    mcmc_chains: int = 2
    confidence_level: float = 0.95
    uncertainty_threshold: float = 0.3
    evidence_weight_prior: str = "beta"  # "beta", "normal", "uniform"
    source_reliability_prior: str = "beta"
    temporal_decay_rate: float = 0.1
    enable_uncertainty_propagation: bool = True


class BayesianVerificationModel:
    """Bayesian model for fact verification with uncertainty quantification."""

    def __init__(self, config: UncertaintyConfig | None = None):
        self.config = config or UncertaintyConfig()
        self.model = None
        self.trace = None
        self.logger = logging.getLogger(__name__)

    def build_model(self, evidence_data: dict[str, Any]):
        """Build Bayesian model for verification."""
        _ensure_bayesian_libraries()  # Lazy initialization
        with pm.Model() as model:
            # Prior for source reliability
            if self.config.source_reliability_prior == "beta":
                source_reliability = pm.Beta(
                    "source_reliability", alpha=2, beta=2)
            else:
                source_reliability = pm.Uniform(
                    "source_reliability", lower=0, upper=1)

            # Prior for evidence weights
            n_evidence = len(evidence_data.get("evidence_scores", []))
            if n_evidence > 0:
                if self.config.evidence_weight_prior == "beta":
                    evidence_weights = pm.Beta(
                        "evidence_weights", alpha=1, beta=1, shape=n_evidence)
                else:
                    evidence_weights = pm.Dirichlet(
                        "evidence_weights", a=np.ones(n_evidence))

            # Temporal decay for evidence
            evidence_ages = evidence_data.get("evidence_ages", [])
            if evidence_ages:
                temporal_factors = pm.Deterministic(
                    "temporal_factors",
                    pm.math.exp(-self.config.temporal_decay_rate *
                                np.array(evidence_ages)),
                )

            # Likelihood for verification outcome
            evidence_scores = evidence_data.get("evidence_scores", [])
            if evidence_scores and n_evidence > 0:
                # Weighted evidence score
                if evidence_ages:
                    weighted_evidence = pm.Deterministic(
                        "weighted_evidence",
                        pm.math.sum(
                            evidence_weights * np.array(evidence_scores) * temporal_factors),
                    )
                else:
                    weighted_evidence = pm.Deterministic(
                        "weighted_evidence",
                        pm.math.sum(evidence_weights *
                                    np.array(evidence_scores)),
                    )

                # Verification probability
                verification_prob = pm.Deterministic(
                    "verification_prob", source_reliability * weighted_evidence)

                # Observed verification outcome (if available)
                observed_outcome = evidence_data.get("observed_outcome")
                if observed_outcome is not None:
                    outcome = pm.Bernoulli(
                        "outcome", p=verification_prob, observed=observed_outcome)
                else:
                    outcome = pm.Bernoulli("outcome", p=verification_prob)

        return model

    def fit_model(self, evidence_data: dict[str, Any]) -> dict[str, Any]:
        """Fit Bayesian model and return posterior samples."""
        _ensure_bayesian_libraries()  # Lazy initialization
        self.model = self.build_model(evidence_data)

        with self.model:
            # Sample from posterior
            self.trace = pm.sample(
                draws=self.config.mcmc_samples,
                tune=self.config.mcmc_tune,
                chains=self.config.mcmc_chains,
                return_inferencedata=True,
                progressbar=False,
            )

        # Extract results
        results = self._extract_results()
        return results

    def _extract_results(self) -> dict[str, Any]:
        """Extract results from MCMC trace."""
        if self.trace is None:
            return {}

        results = {}

        # Verification probability statistics
        if "verification_prob" in self.trace.posterior:
            verification_prob = self.trace.posterior["verification_prob"].values.flatten(
            )
            results["verification_prob"] = {
                "mean": float(np.mean(verification_prob)),
                "std": float(np.std(verification_prob)),
                "median": float(np.median(verification_prob)),
                "credible_interval": self._compute_credible_interval(verification_prob),
                "samples": verification_prob.tolist(),
            }

        # Source reliability statistics
        if "source_reliability" in self.trace.posterior:
            source_reliability = self.trace.posterior["source_reliability"].values.flatten(
            )
            results["source_reliability"] = {
                "mean": float(np.mean(source_reliability)),
                "std": float(np.std(source_reliability)),
                "credible_interval": self._compute_credible_interval(source_reliability),
            }

        # Evidence weights
        if "evidence_weights" in self.trace.posterior:
            evidence_weights = self.trace.posterior["evidence_weights"].values
            results["evidence_weights"] = {
                "mean": np.mean(evidence_weights, axis=(0, 1)).tolist(),
                "std": np.std(evidence_weights, axis=(0, 1)).tolist(),
            }

        # Model diagnostics
        results["diagnostics"] = self._compute_diagnostics()

        return results

    def _compute_credible_interval(self, samples: np.ndarray) -> tuple[float, float]:
        """Compute credible interval for samples."""
        alpha = 1 - self.config.confidence_level
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        return (float(lower), float(upper))

    def _compute_diagnostics(self) -> dict[str, Any]:
        """Compute model diagnostics."""
        if self.trace is None:
            return {}

        try:
            _ensure_bayesian_libraries()  # Lazy initialization
            # R-hat statistic
            rhat = az.rhat(self.trace)

            # Effective sample size
            ess = az.ess(self.trace)

            # MCMC standard error
            mcse = az.mcse(self.trace)

            diagnostics = {
                "rhat": {
                    var: (
                        float(val.values.item())
                        if hasattr(val, "values") and val.values.size == 1
                        else (float(val.values.mean()) if hasattr(val, "values") else float(val))
                    )
                    for var, val in rhat.data_vars.items()
                },
                "ess": {
                    var: (
                        float(val.values.item())
                        if hasattr(val, "values") and val.values.size == 1
                        else (float(val.values.mean()) if hasattr(val, "values") else float(val))
                    )
                    for var, val in ess.data_vars.items()
                },
                "mcse": {
                    var: (
                        float(val.values.item())
                        if hasattr(val, "values") and val.values.size == 1
                        else (float(val.values.mean()) if hasattr(val, "values") else float(val))
                    )
                    for var, val in mcse.data_vars.items()
                },
            }

            return diagnostics

        except Exception as e:
            raise AnalysisError(f"Failed to compute diagnostics: {e}") from e


class UncertaintyPropagationEngine:
    """Engine for propagating uncertainty through verification pipeline."""

    def __init__(self, config: UncertaintyConfig | None = None):
        self.config = config or UncertaintyConfig()
        self.logger = logging.getLogger(__name__)

    def propagate_cluster_uncertainty(self, cluster_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Propagate uncertainty across cluster verification results."""
        if not cluster_results:
            return {"overall_uncertainty": 1.0, "confidence": 0.0}

        # Extract individual uncertainties and confidences
        uncertainties = []
        confidences = []
        weights = []

        for result in cluster_results:
            uncertainty = result.get("uncertainty", 0.5)
            confidence = result.get("confidence", 0.5)
            weight = result.get("evidence_count", 1)

            uncertainties.append(uncertainty)
            confidences.append(confidence)
            weights.append(weight)

        # Weighted uncertainty propagation
        weights = np.array(weights)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum  # Normalize
        else:
            # Equal weights if sum is zero
            weights = np.ones_like(weights) / len(weights)

        # Combine uncertainties using weighted variance
        weighted_uncertainty = np.sqrt(
            np.sum(weights * np.array(uncertainties) ** 2))

        # Combine confidences
        weighted_confidence = np.sum(weights * np.array(confidences))

        # Compute correlation effects (simplified)
        correlation_factor = self._estimate_correlation_factor(cluster_results)
        adjusted_uncertainty = weighted_uncertainty * correlation_factor

        return {
            "overall_uncertainty": float(adjusted_uncertainty),
            "confidence": float(weighted_confidence),
            "individual_uncertainties": uncertainties,
            "correlation_factor": float(correlation_factor),
            "propagation_method": "weighted_variance",
        }

    def _estimate_correlation_factor(self, cluster_results: list[dict[str, Any]]) -> float:
        """Estimate correlation factor between cluster results."""
        if len(cluster_results) <= 1:
            return 1.0

        # Simple heuristic: higher correlation increases overall uncertainty
        # This is a simplified approach; in practice, you'd use more sophisticated methods

        # Check for consistency in verdicts
        verdicts = [result.get("verdict", "UNKNOWN")
                    for result in cluster_results]
        unique_verdicts = set(verdicts)

        if len(unique_verdicts) == 1:
            # All verdicts agree - lower correlation factor
            return 0.8
        elif len(unique_verdicts) == len(verdicts):
            # All verdicts disagree - higher correlation factor
            return 1.3
        else:
            # Mixed agreement - moderate correlation factor
            return 1.0

    def compute_evidence_uncertainty(self, evidence_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute uncertainty for a collection of evidence."""
        if not evidence_list:
            return {"uncertainty": 1.0, "reliability": 0.0}

        # Extract evidence features
        scores = []
        ages = []
        source_reliabilities = []

        for evidence in evidence_list:
            scores.append(evidence.get("score", 0.5))
            ages.append(evidence.get("age_days", 0))
            source_reliabilities.append(
                evidence.get("source_reliability", 0.7))

        # Compute score uncertainty
        score_mean = np.mean(scores)
        score_std = np.std(scores) if len(scores) > 1 else 0.1

        # Temporal uncertainty (older evidence is less reliable)
        max_age = max(ages) if ages else 0
        temporal_uncertainty = 1 - \
            np.exp(-self.config.temporal_decay_rate * max_age)

        # Source reliability uncertainty
        reliability_mean = np.mean(source_reliabilities)
        reliability_std = np.std(source_reliabilities) if len(
            source_reliabilities) > 1 else 0.1

        # Combined uncertainty
        combined_uncertainty = np.sqrt(
            score_std**2 + temporal_uncertainty**2 + reliability_std**2)

        return {
            "uncertainty": float(min(1.0, combined_uncertainty)),
            "reliability": float(reliability_mean),
            "score_uncertainty": float(score_std),
            "temporal_uncertainty": float(temporal_uncertainty),
            "source_uncertainty": float(reliability_std),
            "evidence_count": len(evidence_list),
        }


class BayesianUncertaintyHandler:
    """Main handler for Bayesian uncertainty in fact verification."""

    def __init__(self, config: UncertaintyConfig | None = None):
        self.config = config or UncertaintyConfig()
        self.verification_model = BayesianVerificationModel(config)
        self.propagation_engine = UncertaintyPropagationEngine(config)
        self.logger = logging.getLogger(__name__)

    async def analyze_verification_uncertainty(self, verification_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze uncertainty for a verification task."""
        # Prepare evidence data for Bayesian analysis
        evidence_data = self._prepare_evidence_data(verification_data)

        # Fit Bayesian model
        bayesian_results = self.verification_model.fit_model(evidence_data)

        # Compute evidence uncertainty
        evidence_uncertainty = self.propagation_engine.compute_evidence_uncertainty(
            verification_data.get("evidence", [])
        )

        # Combine results
        uncertainty_analysis = {
            "bayesian_results": bayesian_results,
            "evidence_uncertainty": evidence_uncertainty,
            "overall_confidence": self._compute_overall_confidence(bayesian_results, evidence_uncertainty),
            "uncertainty_level": self._classify_uncertainty_level(bayesian_results, evidence_uncertainty),
            "recommendations": self._generate_recommendations(bayesian_results, evidence_uncertainty),
        }

        return uncertainty_analysis

    def _prepare_evidence_data(self, verification_data: dict[str, Any]) -> dict[str, Any]:
        """Prepare evidence data for Bayesian model."""
        evidence = verification_data.get("evidence", [])

        evidence_scores = []
        evidence_ages = []

        for item in evidence:
            # Extract score (0-1 scale)
            score = item.get("score", 0.5)
            if isinstance(score, str):
                # Convert string scores to numeric
                score_map = {"high": 0.8, "medium": 0.5, "low": 0.2}
                score = score_map.get(score.lower(), 0.5)
            evidence_scores.append(float(score))

            # Extract age in days
            age = item.get("age_days", 0)
            evidence_ages.append(float(age))

        # Observed outcome (if available)
        observed_outcome = verification_data.get("ground_truth")
        if observed_outcome is not None:
            if isinstance(observed_outcome, str):
                outcome_map = {"true": 1, "false": 0, "mixed": 0.5}
                observed_outcome = outcome_map.get(observed_outcome.lower())

        return {
            "evidence_scores": evidence_scores,
            "evidence_ages": evidence_ages,
            "observed_outcome": observed_outcome,
        }

    def _compute_overall_confidence(
        self, bayesian_results: dict[str, Any], evidence_uncertainty: dict[str, Any]
    ) -> float:
        """Compute overall confidence score."""
        # Extract verification probability confidence
        verification_prob = bayesian_results.get("verification_prob", {})
        prob_std = verification_prob.get("std", 0.5)

        # Extract evidence reliability
        evidence_reliability = evidence_uncertainty.get("reliability", 0.5)
        evidence_unc = evidence_uncertainty.get("uncertainty", 0.5)

        # Combine confidences (inverse of uncertainty)
        prob_confidence = 1 - min(1.0, prob_std * 2)  # Scale std to confidence
        evidence_confidence = evidence_reliability * (1 - evidence_unc)

        # Weighted average
        overall_confidence = 0.6 * prob_confidence + 0.4 * evidence_confidence

        return float(max(0.0, min(1.0, overall_confidence)))

    def _classify_uncertainty_level(
        self, bayesian_results: dict[str, Any], evidence_uncertainty: dict[str, Any]
    ) -> str:
        """Classify uncertainty level as low, medium, or high."""
        overall_uncertainty = evidence_uncertainty.get("uncertainty", 0.5)

        verification_prob = bayesian_results.get("verification_prob", {})
        prob_std = verification_prob.get("std", 0.5)

        # Combined uncertainty metric
        combined_uncertainty = (overall_uncertainty + prob_std) / 2

        if combined_uncertainty < self.config.uncertainty_threshold * 0.5:
            return "low"
        elif combined_uncertainty < self.config.uncertainty_threshold:
            return "medium"
        else:
            return "high"

    def _generate_recommendations(
        self, bayesian_results: dict[str, Any], evidence_uncertainty: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on uncertainty analysis."""
        recommendations = []

        # Check verification probability uncertainty
        verification_prob = bayesian_results.get("verification_prob", {})
        prob_std = verification_prob.get("std", 0.5)

        if prob_std > 0.3:
            recommendations.append(
                "High uncertainty in verification probability - consider gathering more evidence")

        # Check evidence count
        evidence_count = evidence_uncertainty.get("evidence_count", 0)
        if evidence_count < 3:
            recommendations.append(
                "Limited evidence available - seek additional sources")

        # Check source reliability
        source_reliability = evidence_uncertainty.get("reliability", 0.5)
        if source_reliability < 0.6:
            recommendations.append(
                "Low source reliability - verify with more credible sources")

        # Check temporal uncertainty
        temporal_unc = evidence_uncertainty.get("temporal_uncertainty", 0.0)
        if temporal_unc > 0.5:
            recommendations.append(
                "Evidence may be outdated - seek more recent information")

        # Check model diagnostics
        diagnostics = bayesian_results.get("diagnostics", {})

        return recommendations

    def get_uncertainty_stats(self) -> dict[str, Any]:
        """Get statistics about the uncertainty handler."""
        return {
            "config": {
                "use_bayesian_inference": self.config.use_bayesian_inference,
                "mcmc_samples": self.config.mcmc_samples,
                "confidence_level": self.config.confidence_level,
                "uncertainty_threshold": self.config.uncertainty_threshold,
            },
            "model_initialized": self.verification_model.model is not None,
            "last_trace_available": self.verification_model.trace is not None,
        }
