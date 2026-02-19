from enum import StrEnum


class EnvironmentType(StrEnum):
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"


class AgentType(StrEnum):
    JOB_SELECTOR = "job_selector"
    AGV_SELECTOR = "agv_selector"


class ObservationType(StrEnum):
    DICT = "dict"
    FLAT = "flat"
    GRAPH = "graph"
    GRAPH_MATRIX = "graph_matrix"


class JobSelectorType(StrEnum):
    JOB = "job"
    OPERATION = "operation"


class MarlEnvType(StrEnum):
    AEC = "aec"
    PARALLEL = "parallel"


class ItemDataType(StrEnum):
    AGV = "agv"
    JOB = "job"
    OP = "operation"


class JobAgentType(StrEnum):
    POLICY = "policy"
    HEURISTIC = "heuristic"


class AgvAgentType(StrEnum):
    POLICY = "policy"
    HEURISTIC = "heuristic"


class GraphNormalization(StrEnum):
    NONE = "none"
    JOB = "job"
    OPERATION = "operation"
