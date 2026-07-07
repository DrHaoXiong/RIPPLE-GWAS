"""Local weak-signal module discovery."""

from ripple.modules.anchored import (
    AnchoredModuleLibrary,
    anchored_module_tests,
    build_louvain_anchor_library,
    gene_sets_to_library,
    load_anchored_gene_set_library,
    merge_anchored_libraries,
    render_anchored_module_report,
)
from ripple.modules.discovery import (
    DEFAULT_DR_GENE_SETS,
    calibrate_local_modules,
    discover_local_modules,
    load_gene_sets,
    pathway_subgraph_tests,
    render_module_discovery_report,
    run_local_module_discovery,
    selection_aware_module_null,
)
from ripple.modules.distributed import (
    RippleDConfig,
    assign_pseudo_loci,
    classify_distributed_module,
    contribution_metrics,
    locus_robust_stat,
    ripple_d_module_tests,
    ripple_d_stat,
    summarize_module_distribution,
)

__all__ = [
    "AnchoredModuleLibrary",
    "DEFAULT_DR_GENE_SETS",
    "RippleDConfig",
    "anchored_module_tests",
    "assign_pseudo_loci",
    "build_louvain_anchor_library",
    "calibrate_local_modules",
    "classify_distributed_module",
    "contribution_metrics",
    "discover_local_modules",
    "gene_sets_to_library",
    "load_anchored_gene_set_library",
    "load_gene_sets",
    "locus_robust_stat",
    "merge_anchored_libraries",
    "pathway_subgraph_tests",
    "render_anchored_module_report",
    "render_module_discovery_report",
    "ripple_d_module_tests",
    "ripple_d_stat",
    "run_local_module_discovery",
    "selection_aware_module_null",
    "summarize_module_distribution",
]
