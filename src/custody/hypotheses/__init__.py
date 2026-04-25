"""Hypothesis layer — uncertainty-to-tasking engine (ADR-0021).

Slice 1 exposes the contract types, scenario registry, deterministic
belief update, and human-readable explain helpers.  This layer sits
alongside (not inside) the seven-layer entity reasoning stack: it answers
"what do we believe is happening in this scenario across competing
explanations?" while ``belief_assessment.FusionAssessment`` continues to
answer "what do we believe about this entity right now?"
"""

from custody.hypotheses.collection_value import (
    CollectionCandidate,
    CollectionRecommendation,
    CollectionValue,
    rank_collection_candidates,
)
from custody.hypotheses.counterfactual import (
    CounterfactualCollectAssessment,
    CounterfactualOutcome,
    CounterfactualReport,
    format_counterfactual_text,
    simulate_counterfactual_collects,
)
from custody.hypotheses.custody_health import (
    CustodyHealthStatus,
    HypothesisCustodyHealth,
    assess_custody_health,
)
from custody.hypotheses.decision_packet import (
    BeliefEntry,
    CollectRecommendation,
    DecisionPacket,
    SCHEMA_VERSION,
    build_decision_packet,
    format_as_json,
    format_as_markdown,
    format_as_text,
    packet_to_json_object,
)
from custody.hypotheses.evidence import (
    from_mapping,
    from_match,
    from_observation,
    from_scene,
)
from custody.hypotheses.explain import explain_top, format_state
from custody.hypotheses.mission_value import (
    MissionValueAssessment,
    MissionValueComponent,
    MissionValueReport,
    attribute_mission_value,
    format_mission_value_markdown,
    format_mission_value_text,
    mission_value_to_json_object,
)
from custody.hypotheses.optimizer import (
    OptimizedPlan,
    PlanConstraint,
    PlanItem,
    PlanOptimizationReport,
    format_optimizer_text,
    optimize_collection_plan,
)
from custody.hypotheses.planner_queue import (
    PlannerQueue,
    PlannerQueueItem,
    PlannerQueueStatus,
    build_planner_queue,
    create_queue_item,
    format_queue_json,
    format_queue_markdown,
    format_queue_text,
    load_review_ledger_jsonl,
    record_status_from_review,
    select_latest_review_for_scenario,
)
from custody.hypotheses.planner_review import (
    PlannerReviewRecord,
    ReviewAction,
    ReviewLedgerAppendResult,
    ReviewSubject,
    append_review_record_jsonl,
    create_planner_review_record,
    create_review_subject_from_optimized_plan,
    create_review_subject_from_policy_evaluation,
    format_review_markdown,
    format_review_text,
    record_to_dict,
    record_to_json,
)
from custody.hypotheses.policy_eval import (
    PolicyDefinition,
    PolicyEvaluation,
    PolicyEvaluationReport,
    evaluate_collection_policies,
    format_policy_eval_text,
)
from custody.hypotheses.registry import (
    SCENARIO_TENNENT,
    SCENARIO_WHITSUN,
    get_hypotheses,
    get_hypothesis_ids,
)
from custody.hypotheses.types import (
    Hypothesis,
    HypothesisEvidence,
    HypothesisState,
)
from custody.hypotheses.update import update_state

__all__ = [
    "BeliefEntry",
    "CollectRecommendation",
    "CollectionCandidate",
    "CollectionRecommendation",
    "CollectionValue",
    "CounterfactualCollectAssessment",
    "CounterfactualOutcome",
    "CounterfactualReport",
    "CustodyHealthStatus",
    "DecisionPacket",
    "Hypothesis",
    "HypothesisCustodyHealth",
    "HypothesisEvidence",
    "HypothesisState",
    "MissionValueAssessment",
    "MissionValueComponent",
    "MissionValueReport",
    "OptimizedPlan",
    "PlanConstraint",
    "PlanItem",
    "PlanOptimizationReport",
    "PlannerQueue",
    "PlannerQueueItem",
    "PlannerQueueStatus",
    "PlannerReviewRecord",
    "PolicyDefinition",
    "PolicyEvaluation",
    "PolicyEvaluationReport",
    "ReviewAction",
    "ReviewLedgerAppendResult",
    "ReviewSubject",
    "SCENARIO_TENNENT",
    "SCENARIO_WHITSUN",
    "SCHEMA_VERSION",
    "attribute_mission_value",
    "append_review_record_jsonl",
    "assess_custody_health",
    "build_decision_packet",
    "build_planner_queue",
    "create_planner_review_record",
    "create_queue_item",
    "create_review_subject_from_optimized_plan",
    "create_review_subject_from_policy_evaluation",
    "evaluate_collection_policies",
    "explain_top",
    "format_as_json",
    "format_as_markdown",
    "format_as_text",
    "format_counterfactual_text",
    "format_mission_value_markdown",
    "format_optimizer_text",
    "format_mission_value_text",
    "format_policy_eval_text",
    "format_queue_json",
    "format_queue_markdown",
    "format_queue_text",
    "format_review_markdown",
    "format_review_text",
    "format_state",
    "from_mapping",
    "from_match",
    "from_observation",
    "from_scene",
    "get_hypotheses",
    "get_hypothesis_ids",
    "load_review_ledger_jsonl",
    "mission_value_to_json_object",
    "optimize_collection_plan",
    "packet_to_json_object",
    "rank_collection_candidates",
    "record_status_from_review",
    "record_to_dict",
    "record_to_json",
    "select_latest_review_for_scenario",
    "simulate_counterfactual_collects",
    "update_state",
]
