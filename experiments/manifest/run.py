"""Aggregate every quotable number into ``results/manifest.json``.

One flat namespace of stable dotted keys.  Every number quoted in the paper is
read from this file, and can be traced by its key.

Keys are grouped by where they come from:
  benchmark.*        the published values we are compared against
  site.*             NESCA geometry and environment
  data.*             the tephra dataset
  forward.*          forward-model verification
  identify.*         what the deposit can resolve
  synthetic.*        validation on known truths
  nesca.*            the inversion result
  sources.*          source discrimination
  modes.*            singular modes, resolution kernels and Laplace abscissa
  shape.*            clast shape (also the continuous shape plane)
  kernel_maps.*      the front-limited kernel in space and time
  nesca_vent.*       the NESCA posterior with the vent sampled (Fig. 3 run)
  vent.*             vent position (also the vent misfit surface)
  context.*          the NESCA heat flux against other heat sources

Keys read from the newer results files (the tables below ``main``) also carry
their source path in ``_sources`` as ``results/<file>.json#<path>``.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = R / "manifest.json"


def load(name):
    return json.loads((R / name).read_text())


def at(doc, path: str):
    """Follow a dotted ``path`` through nested JSON.

    List items are addressed as ``[i]`` or as a bare integer segment.  Dict keys
    may themselves contain dots (``x_for_p_0.95``), so at each level the longest
    key that matches the front of the remaining path wins.
    """
    node, rest = doc, path
    while rest:
        if rest.startswith("["):
            j = rest.index("]")
            node = node[int(rest[1:j])]
            rest = rest[j + 1 :].removeprefix(".")
        elif isinstance(node, list):
            seg, _, rest = rest.partition(".")
            node = node[int(seg)]
        else:
            for k in sorted(node, key=len, reverse=True):
                if rest == k or rest.startswith(k + ".") or rest.startswith(k + "["):
                    node, rest = node[k], rest[len(k) :].removeprefix(".")
                    break
            else:
                raise KeyError(f"{path}: no key matches {rest!r}")
    return node


def lohi(d):
    """A ``{"low": a, "high": b}`` range as the list ``[a, b]``."""
    return [d["low"], d["high"]]


# Keys read by path from the newer results files: (key, file, path[, transform]).
PICKED = [
    # ---- model constants quoted in Methods (results/constants.json) ----
    ("constants.rho_tephra_kgm3", "constants.json", "rho_tephra_kgm3"),
    ("constants.nu_seawater_m2s", "constants.json", "nu_seawater_m2s"),
    ("constants.epsilon_entrainment", "constants.json", "epsilon_entrainment"),
    ("constants.lambda_front", "constants.json", "lambda_front"),
    ("constants.rho_seawater_kgm3", "constants.json", "rho_seawater_kgm3"),
    ("constants.c_p_seawater", "constants.json", "c_p_seawater_J_kg_K"),
    ("constants.alpha_thermal_expansion", "constants.json", "alpha_thermal_expansion_K"),
    ("constants.c_f_point", "constants.json", "c_f_point_closure"),
    ("forward.mtt_point_zmax_coefficient", "constants.json", "mtt_point_zmax_coefficient"),
    ("forward.mtt_point_zneutral_coefficient", "constants.json", "mtt_point_zneutral_coefficient"),
    ("forward.mtt_planar_zmax_coefficient", "constants.json", "mtt_planar_zmax_coefficient"),
    ("site.rise_f0_m4s3", "constants.json", "rise_height_buoyancy_flux_m4s3"),
    # ---- modes of the NESCA forward operator (Fig. 2) ----
    ("modes.reproduces_identifiability", "modes.json", "verification.reproduces_identifiability"),
    ("modes.nu0_kg", "modes.json", "construction.nu0_kg"),
    ("modes.v1_peak_log10q", "modes.json", "right_singular_vectors.per_mode.0.peak_log10Q"),
    ("modes.v2_peak_log10q", "modes.json", "right_singular_vectors.per_mode.1.peak_log10Q"),
    ("modes.v3_peak_log10q", "modes.json", "right_singular_vectors.per_mode.2.peak_log10Q"),
    ("modes.v6_peak_log10q", "modes.json", "right_singular_vectors.per_mode.5.peak_log10Q"),
    ("modes.q_mean_log10", "modes.json", "q_mass_weighted_mean_m3s.log10"),
    (
        "modes.kernel_width_one_fraction_log10q",
        "modes.json",
        "resolution.one_fraction_snr.fwhm_log10Q",
    ),
    (
        "modes.kernel_width_one_fraction_picard_log10q",
        "modes.json",
        "resolution.one_fraction_picard.fwhm_log10Q",
    ),
    (
        "modes.kernel_width_four_fractions_log10q",
        "modes.json",
        "resolution.four_fractions_snr.fwhm_log10Q",
    ),
    (
        "modes.kernel_width_four_fractions_picard_log10q",
        "modes.json",
        "resolution.four_fractions_picard.fwhm_log10Q",
    ),
    (
        "modes.kernel_peak_offset_one_fraction_log10q",
        "modes.json",
        "resolution.one_fraction_snr.peak_offset_from_target_log10Q",
    ),
    (
        "modes.kernel_peak_offset_four_fractions_log10q",
        "modes.json",
        "resolution.four_fractions_snr.peak_offset_from_target_log10Q",
    ),
    (
        "modes.kernel_peak_offset_picard_log10q",
        "modes.json",
        "resolution.four_fractions_picard.peak_offset_from_target_log10Q",
    ),
    (
        "modes.resolved_window_one_fraction_log10q",
        "modes.json",
        "resolution.resolved_window_log10Q_5_95.one_fraction_snr",
    ),
    (
        "modes.resolved_window_four_fractions_log10q",
        "modes.json",
        "resolution.resolved_window_log10Q_5_95.four_fractions_snr",
    ),
    (
        "modes.resolved_window_picard_log10q",
        "modes.json",
        "resolution.resolved_window_log10Q_5_95.four_fractions_picard",
    ),
    (
        "modes.lambda_factor_one_fraction",
        "modes.json",
        "laplace_abscissa.one_fraction_250_500.factor",
    ),
    ("modes.lambda_factor_four_fractions", "modes.json", "laplace_abscissa.four_fractions.factor"),
    (
        "modes.lambda_decades_one_fraction",
        "modes.json",
        "laplace_abscissa.one_fraction_250_500.decades",
    ),
    (
        "modes.lambda_decades_four_fractions",
        "modes.json",
        "laplace_abscissa.four_fractions.decades",
    ),
    (
        "modes.w_ratio_four_fractions",
        "modes.json",
        "laplace_abscissa.four_fractions.w_max_over_w_min",
    ),
    ("modes.core_radius_min_m", "modes.json", "laplace_abscissa.per_class.250-500um.r_min_m"),
    ("modes.core_radius_max_m", "modes.json", "laplace_abscissa.per_class.250-500um.r_max_m"),
    (
        "modes.sign_changes_seen_one_fraction",
        "modes.json",
        "data_space_patterns.zero_crossings_inside_sampled_range.one_fraction_250_500",
    ),
    (
        "modes.sign_changes_seen_four_fractions",
        "modes.json",
        "data_space_patterns.zero_crossings_inside_sampled_range.four_fractions",
    ),
    ("modes.data_norm_first2", "modes.json", "data_norm_captured.four_fractions.first_2"),
    ("modes.data_norm_first6", "modes.json", "data_norm_captured.four_fractions.first_6"),
    (
        "modes.data_norm_in_range",
        "modes.json",
        "data_norm_captured.four_fractions_range_of_operator",
    ),
    (
        "modes.halving_gain_one_fraction",
        "modes.json",
        "noise_vs_fractions.gain_from_halving_noise_one_fraction",
    ),
    (
        "modes.halving_gain_four_fractions",
        "modes.json",
        "noise_vs_fractions.gain_from_halving_noise_four_fractions",
    ),
    (
        "modes.fraction_gain_at_nugget",
        "modes.json",
        "noise_vs_fractions.gain_from_three_fractions_at_nugget",
    ),
    # ---- clast shape over the continuous shape plane (Fig. 5) ----
    (
        "shape.dstar_best_um",
        "shape_space.json",
        "length_scale_test.continuous.best_fit.transition_diameter_um",
    ),
    (
        "shape.c1sq_over_c2_best",
        "shape_space.json",
        "length_scale_test.continuous.best_fit.c1sq_over_c2",
    ),
    (
        "shape.chi2_min_continuous",
        "shape_space.json",
        "length_scale_test.continuous.best_fit.chi2_min",
    ),
    (
        "shape.dstar_ci68_um",
        "shape_space.json",
        "length_scale_test.continuous.intervals_one_parameter.dchi2_1.transition_diameter_um",
    ),
    (
        "shape.dstar_ci95_um",
        "shape_space.json",
        "length_scale_test.continuous.intervals_one_parameter.dchi2_4.transition_diameter_um",
    ),
    ("shape.dstar_sheet_um", "shape_space.json", "settling_curves.transition_diameter_um.sheet"),
    ("shape.dstar_blocky_um", "shape_space.json", "settling_curves.transition_diameter_um.blocky"),
    ("shape.dstar_long_um", "shape_space.json", "settling_curves.transition_diameter_um.long"),
    ("shape.boot_dstar_ci95_um", "shape_space.json", "bootstrap.transition_diameter_um.ci95"),
    (
        "shape.phi_valley_c1_15_W",
        "shape_space.json",
        "length_scale_test.heat_flux.valley.phi_at_c1_15_W",
    ),
    (
        "shape.phi_valley_c1_45_W",
        "shape_space.json",
        "length_scale_test.heat_flux.valley.phi_at_c1_45_W",
    ),
    (
        "shape.phi_marginal_continuous_W",
        "shape_space.json",
        "length_scale_test.heat_flux.marginal_continuous.nominal_box."
        "phi_with_within_shape_scatter_W.median",
    ),
    (
        "shape.phi_marginal_continuous_ci95_W",
        "shape_space.json",
        "length_scale_test.heat_flux.marginal_continuous.nominal_box."
        "phi_with_within_shape_scatter_W.ci95",
    ),
    (
        "shape.nesca_nu_blocky_ratio_noise_free",
        "shape_space.json",
        "synthetic.noise_free.nesca_nu.blocky.extreme_ratio",
    ),
    (
        "shape.lengthscale_confusion_blocky_as_sheet",
        "shape_space.json",
        "synthetic.main.nesca_nu.sigma_residual.length_scale_test.volcaniclast_candidates."
        "confusion_fraction[0][2]",
    ),
    (
        "shape.lengthscale_p_correct_single_flux",
        "shape_space.json",
        "synthetic.main.steady.sigma_residual.length_scale_test.volcaniclast_candidates."
        "p_correct_mean",
    ),
    (
        "shape.profile_p_correct_nesca",
        "shape_space.json",
        "synthetic.main.nesca_nu.sigma_residual.profile_test.volcaniclast_candidates."
        "p_correct_mean",
    ),
    (
        "shape.profile_dchi2_sheet",
        "shape_space.json",
        "profile_test.real_deposit.by_shape.sheet.delta_chi2",
    ),
    (
        "shape.profile_dchi2_blocky",
        "shape_space.json",
        "profile_test.real_deposit.by_shape.blocky.delta_chi2",
    ),
    (
        "shape.profile_log_q_sd_blocky",
        "shape_space.json",
        "profile_test.real_deposit.by_shape.blocky.best_log_q_sd",
    ),
    (
        "shape.profile_single_flux_penalty_blocky",
        "shape_space.json",
        "profile_test.real_deposit.by_shape.blocky.delta_chi2_single_flux_minus_best",
    ),
    (
        "shape.profile_weight_blocky",
        "shape_space.json",
        "profile_test.real_deposit.weights_volcaniclast.blocky",
    ),
    (
        "shape.profile_weight_long",
        "shape_space.json",
        "profile_test.real_deposit.weights_volcaniclast.long",
    ),
    (
        "shape.profile_weight_sheet",
        "shape_space.json",
        "profile_test.real_deposit.weights_volcaniclast.sheet",
    ),
    (
        "shape.profile_phi_marginal_W",
        "shape_space.json",
        "profile_test.heat_flux_three_class_mixture.phi_W.median",
    ),
    (
        "shape.profile_phi_marginal_ci95_W",
        "shape_space.json",
        "profile_test.heat_flux_three_class_mixture.phi_W.ci95",
    ),
    (
        "shape.profile_dstar_best_um",
        "shape_space.json",
        "profile_test.continuous.best_transition_diameter_um",
    ),
    (
        "shape.profile_dstar_ci95_um",
        "shape_space.json",
        "profile_test.continuous.intervals_one_parameter.dchi2_4.transition_diameter_um",
    ),
    (
        "shape.power_info_for_p95",
        "shape_space.json",
        "synthetic.power.profile_test_nesca_nu.information_collapse.x_for_p_0.95",
    ),
    (
        "shape.power_ncores_p95_sigma_residual",
        "shape_space.json",
        "synthetic.power.profile_test_nesca_nu.information_collapse."
        "n_cores_for_p_0.95_at_sigma_residual",
    ),
    (
        "shape.power_sigma_p95_124_cores",
        "shape_space.json",
        "synthetic.power.profile_test_nesca_nu.information_collapse.sigma_for_p_0.95_at_124_cores",
    ),
    (
        "shape.residual_correlation_fractions",
        "shape_space.json",
        "real_data.residual_correlation_across_fractions.mean_off_diagonal",
    ),
    # ---- the front-limited kernel in space and time (Fig. 1) ----
    ("kernel_maps.cored_r_min_m", "kernel_maps.json", "cored_radii.r_min_m"),
    ("kernel_maps.cored_r_max_m", "kernel_maps.json", "cored_radii.r_max_m"),
    (
        "kernel_maps.front_radius_at_shutoff_m",
        "kernel_maps.json",
        "front.front_radius_at_shutoff_m",
    ),
    ("kernel_maps.front_t_reaches_r_max_h", "kernel_maps.json", "front.t_reaches_r_max_h"),
    (
        "kernel_maps.front_crossing_over_duration",
        "kernel_maps.json",
        "front.crossing_time_over_duration",
    ),
    (
        "kernel_maps.floor_frac_front_rmax_63_125",
        "kernel_maps.json",
        "deposition_timing.63-125.fraction_on_seafloor_when_front_reaches_r_max",
    ),
    (
        "kernel_maps.floor_frac_front_rmax_125_250",
        "kernel_maps.json",
        "deposition_timing.125-250.fraction_on_seafloor_when_front_reaches_r_max",
    ),
    (
        "kernel_maps.floor_frac_front_rmax_250_500",
        "kernel_maps.json",
        "deposition_timing.250-500.fraction_on_seafloor_when_front_reaches_r_max",
    ),
    (
        "kernel_maps.floor_frac_front_rmax_over500",
        "kernel_maps.json",
        "deposition_timing.500-1000.fraction_on_seafloor_when_front_reaches_r_max",
    ),
    (
        "kernel_maps.floor_frac_shutoff_250_500",
        "kernel_maps.json",
        "deposition_timing.250-500.fraction_on_seafloor_at_shutoff",
    ),
    (
        "kernel_maps.t_half_local_250_500_r_max_h",
        "kernel_maps.json",
        "deposition_timing.250-500.t_half_local_deposit_h.r_max",
    ),
    (
        "kernel_maps.after_shutoff_share_250_500_r_max",
        "kernel_maps.json",
        "deposition_timing.250-500.local_deposit_fraction_landing_after_shutoff.r_max",
    ),
    (
        "kernel_maps.ratio_one_crossing_63_125_m",
        "kernel_maps.json",
        "ratio_map.per_fraction.63-125.ratio_one_crossings_m[0]",
    ),
    (
        "kernel_maps.ratio_one_crossing_125_250_m",
        "kernel_maps.json",
        "ratio_map.per_fraction.125-250.ratio_one_crossings_m[0]",
    ),
    (
        "kernel_maps.ratio_one_crossing_250_500_m",
        "kernel_maps.json",
        "ratio_map.per_fraction.250-500.ratio_one_crossings_m[0]",
    ),
    (
        "kernel_maps.ratio_one_crossings_over500_m",
        "kernel_maps.json",
        "ratio_map.per_fraction.500-1000.ratio_one_crossings_m",
    ),
    (
        "kernel_maps.first_crossing_over_L_min",
        "kernel_maps.json",
        "ratio_map.first_crossing_over_L_min",
    ),
    (
        "kernel_maps.first_crossing_over_L_max",
        "kernel_maps.json",
        "ratio_map.first_crossing_over_L_max",
    ),
    (
        "kernel_maps.ratio_250_500_at_1km",
        "kernel_maps.json",
        "ratio_map.per_fraction.250-500.ratio_at_1km",
    ),
    (
        "kernel_maps.ratio_250_500_at_8km",
        "kernel_maps.json",
        "ratio_map.per_fraction.250-500.ratio_at_8km",
    ),
    (
        "kernel_maps.reproduction_max_rel_diff",
        "kernel_maps.json",
        "reproduction_check.max_abs_rel_difference",
    ),
    # ---- NESCA with the vent sampled: the run drawn in Fig. 3 ----
    ("nesca_vent.mass_total_kg", "nesca_maps.json", "posterior.mass_total_kg.median"),
    ("nesca_vent.mass_total_ci95_kg", "nesca_maps.json", "posterior.mass_total_kg.ci95"),
    ("nesca_vent.q_mean_m3s", "nesca_maps.json", "posterior.q_mass_weighted_mean_m^3s^-1.median"),
    (
        "nesca_vent.q_mean_ci95_m3s",
        "nesca_maps.json",
        "posterior.q_mass_weighted_mean_m^3s^-1.ci95",
    ),
    ("nesca_vent.phi_W", "nesca_maps.json", "posterior.phi_at_mass_weighted_mean_W.median"),
    ("nesca_vent.phi_ci95_W", "nesca_maps.json", "posterior.phi_at_mass_weighted_mean_W.ci95"),
    (
        "nesca_vent.phi_ci95_upper_mcse_W",
        "nesca_maps.json",
        "reproduction_vs_vent_shape._per_chain_quantiles.phi_at_mass_weighted_mean_W.mcse_q975",
    ),
    ("nesca_vent.sigma_log", "nesca_maps.json", "posterior.sigma_log.median"),
    ("nesca_vent.vent_x_median_m", "nesca_maps.json", "posterior.vent_x_m.median"),
    ("nesca_vent.vent_y_median_m", "nesca_maps.json", "posterior.vent_y_m.median"),
    ("nesca_vent.max_rhat", "nesca_maps.json", "diagnostics.max_rhat"),
    ("nesca_vent.n_divergences", "nesca_maps.json", "diagnostics.n_divergences"),
    ("nesca_vent.n_observations", "nesca_maps.json", "n_observations"),
    (
        "nesca_vent.ppc_classes_with_trend",
        "nesca_maps.json",
        "ppc_classes_with_systematic_radial_trend",
    ),
    (
        "nesca_vent.corr_mass_qmean_spearman",
        "nesca_maps.json",
        "joint_mass_qmean.spearman_rho_mass_qmean",
    ),
    (
        "nesca_vent.corr_log_mass_log_qmean_pearson",
        "nesca_maps.json",
        "joint_mass_qmean.pearson_r_log_mass_log_qmean",
    ),
    ("nesca_vent.n_map_draws", "nesca_maps.json", "maps.n_draws"),
    # ---- the vent misfit surface (Fig. 4) ----
    ("vent.misfit_delta_nll_centroid", "vent_misfit.json", "delta_nll_at.survey_centroid"),
    (
        "vent.misfit_delta_nll_profiled",
        "vent_misfit.json",
        "delta_nll_at.least_squares_profiled_vent",
    ),
    ("vent.misfit_argmin_xy_m", "vent_misfit.json", "zoom_grid.argmin_xy_m"),
    ("vent.conditional_over_marginal_sd", "vent_misfit.json", "conditional_over_marginal_sd"),
    ("vent.phi_ci95_width_profiled_W", "vent_misfit.json", "phi_ci95_widths.profiled_W"),
    ("vent.phi_ci95_width_sampled_W", "vent_misfit.json", "phi_ci95_widths.sampled_W"),
    (
        "vent.phi_ci95_width_ratio",
        "vent_misfit.json",
        "phi_ci95_widths.ratio_sampled_over_profiled",
    ),
    # ---- the NESCA heat flux against other heat sources (Fig. 6) ----
    (
        "context.nesca_blocky_over_global_hydrothermal",
        "heat_context.json",
        "ratios.blocky.over_global_hydrothermal_central",
    ),
    (
        "context.nesca_sheet_over_global_hydrothermal",
        "heat_context.json",
        "ratios.sheet.over_global_hydrothermal_central",
    ),
    (
        "context.nesca_blocky_over_ridge_axis",
        "heat_context.json",
        "ratios.blocky.over_global_ridge_axis",
        lohi,
    ),
    (
        "context.nesca_sheet_over_ridge_axis",
        "heat_context.json",
        "ratios.sheet.over_global_ridge_axis",
        lohi,
    ),
    (
        "context.nesca_blocky_over_large_vent_field",
        "heat_context.json",
        "ratios.blocky.over_large_vent_field",
    ),
    (
        "context.nesca_sheet_over_large_vent_field",
        "heat_context.json",
        "ratios.sheet.over_large_vent_field",
    ),
    (
        "context.nesca_blocky_black_smoker_equivalents",
        "heat_context.json",
        "ratios.blocky.mean_black_smokers",
    ),
    ("context.nesca_blocky_over_earth", "heat_context.json", "ratios.blocky.over_whole_earth"),
    (
        "context.nesca_blocky_over_ep86",
        "heat_context.json",
        "ratios.blocky.over_ep86_megaplume",
        lohi,
    ),
    (
        "context.nesca_blocky_hours_for_megaplume_range",
        "heat_context.json",
        "ratios.blocky.hours_to_release_megaplume_range",
    ),
    (
        "context.nesca_blocky_heat_in_reference_duration_J",
        "heat_context.json",
        "ratios.blocky.heat_in_reference_duration_J",
    ),
    (
        "context.nesca_sheet_heat_in_reference_duration_J",
        "heat_context.json",
        "ratios.sheet.heat_in_reference_duration_J",
    ),
    (
        "context.vent_field_days_for_floor",
        "heat_context.json",
        "vent_field_time_for_one_megaplume.floor_days",
    ),
    (
        "context.vent_field_years_for_floor",
        "heat_context.json",
        "vent_field_time_for_one_megaplume.floor_years",
    ),
    (
        "context.vent_field_years_for_ceiling",
        "heat_context.json",
        "vent_field_time_for_one_megaplume.ceiling_years",
    ),
    (
        "context.megaplume_power_at_15h_W",
        "heat_context.json",
        "megaplume_mean_power.mean_power_at_reference_W",
    ),
    ("context.observed_megaplume_heat_J", "heat_context.json", "observed_event_heat.megaplumes_J"),
    (
        "context.observed_event_plume_heat_J",
        "heat_context.json",
        "observed_event_heat.all_event_plumes_J",
    ),
    (
        "context.ep86_chimney_years",
        "heat_context.json",
        "cross_checks.ep86_chimney_years_at_mean_smoker",
    ),
    # ---- source discrimination maps (Fig. 6) ----
    (
        "sources.lava_crossing_duration_direct_h",
        "heat_context.json",
        "lava_map.crossing_duration_h_at_mapped_area",
    ),
    (
        "sources.lava_area_factor_at_reference",
        "heat_context.json",
        "lava_map.area_factor_needed_at_reference_duration",
    ),
    (
        "sources.lava_area_needed_at_reference_km2",
        "heat_context.json",
        "lava_map.area_needed_at_reference_duration_km2",
    ),
    (
        "sources.dyke_plausible_energy_over_floor",
        "heat_context.json",
        "dyke_map.plausible_dyke.energy_over_floor",
    ),
    (
        "sources.dyke_plausible_height_needed_m",
        "heat_context.json",
        "dyke_map.plausible_dyke.height_needed_at_this_length_m",
    ),
    (
        "sources.dyke_plausible_duration_needed_h",
        "heat_context.json",
        "dyke_map.plausible_dyke.duration_needed_for_floor_h",
    ),
    (
        "sources.dyke_plausible_phi_15h_W",
        "heat_context.json",
        "dyke_map.plausible_dyke.phi_at_reference_duration_W",
    ),
    (
        "sources.dyke_face_area_product_needed_m2",
        "heat_context.json",
        "dyke_map.face_area_product_needed_m2",
    ),
    (
        "sources.dyke_fraction_of_prior_box_below_floor",
        "heat_context.json",
        "dyke_map.fraction_of_prior_box_below_floor",
    ),
]


def picked_keys(m: dict) -> dict:
    """Add the PICKED keys (and a few built from several entries) to ``m``.

    Returns the ``key -> results/<file>#<path>`` provenance of every key added.
    """
    docs = {name: load(name) for name in sorted({row[1] for row in PICKED})}
    src = {}

    def put(key, value, source):
        if key in m:
            raise KeyError(f"manifest key {key} is already defined")
        m[key] = value
        src[key] = source

    for key, name, path, *tf in PICKED:
        value = at(docs[name], path)
        put(key, tf[0](value) if tf else value, f"results/{name}#{path}")

    mo = docs["modes.json"]
    for which, field in (("picard", "N_res_picard"), ("snr", "N_res_snr")):
        put(
            f"modes.{which}_by_fraction_count",
            [at(mo, f"verification.by_fraction_count.k={k}.{field}") for k in (1, 2, 3, 4)],
            f"results/modes.json#verification.by_fraction_count.k={{1,2,3,4}}.{field}",
        )
    i60 = at(mo, "modes_vs_n_cores.n_cores").index(60)
    put(
        "modes.modes_four_fractions_60_cores_mean",
        at(mo, f"modes_vs_n_cores.mean_N_res_snr.k=4[{i60}]"),
        f"results/modes.json#modes_vs_n_cores.mean_N_res_snr.k=4[{i60}]",
    )

    # the largest radial residual trend over the four fractions
    ppc = at(docs["nesca_maps.json"], "posterior_predictive_by_class")
    cls = max(ppc, key=lambda c: abs(ppc[c]["radial_slope_z"]))
    put(
        "nesca_vent.ppc_max_abs_slope_z",
        abs(ppc[cls]["radial_slope_z"]),
        f"results/nesca_maps.json#posterior_predictive_by_class.{cls}.radial_slope_z",
    )
    return src


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "uncommitted"


def main() -> int:
    strat, fwd = load("stratification.json"), load("forward_checks.json")
    ident, syn = load("identifiability.json"), load("synthetic.json")
    steady, uns = load("nesca_steady.json"), load("nesca_unsteady.json")
    sbc, mc = load("sbc.json"), load("model_comparison.json")
    shape = load("clast_shape.json")
    env = load("environment.json")
    vs = load("vent_shape.json")
    p = uns["posterior"]
    vb = vs["by_shape"]["blocky"]
    vm = vs["shape_marginal"]
    g = steady["by_shape"]
    dt = mc["delivery_test"]
    ln10 = 2.302585092994046
    ev = mc["evidence"]
    best = max(
        (m for m in ev if "logz" in ev[m].get("nominal", {})),
        key=lambda m: ev[m]["nominal"]["logz"],
    )

    def b10(m):
        return (ev[m]["nominal"]["logz"] - ev[best]["nominal"]["logz"]) / ln10

    m = {
        "_description": "Every quotable number for the PLUME paper.",
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/manifest/run.py",
        "_git_commit": git_hash(),
        "_rule": (
            "Every number quoted in the paper is read from this file and can be traced by its key."
        ),
        "_environment": {
            "python": env["python"],
            "numpyro": env["packages"]["numpyro"],
            "dynesty": env["packages"]["dynesty"],
        },
        # ---- the published benchmark ----
        "benchmark.L_m": 4900.0,
        "benchmark.L_sigma_m": 400.0,
        "benchmark.q_umb_m3s": 7.6e5,
        "benchmark.q_umb_sigma_m3s": 3.6e5,
        "benchmark.phi_W": 1.5e12,
        "benchmark.phi_sigma_W": 0.9e12,
        "benchmark.w_s_assumed_ms": 0.03,
        "benchmark.citation": "pegler2021rapid",
        # ---- site and environment ----
        "site.water_depth_min_m": 3212.0,
        "site.water_depth_max_m": 3371.0,
        "site.water_depth_mean_m": 3303.0,
        "site.N_buoyancy_s": strat["N_primary"]["value_s^-1"],
        "site.N_buoyancy_sigma_s": strat["N_primary"]["sigma_s^-1"],
        "site.N_ratio_to_benchmark": strat["benchmark_comparison"]["ratio_primary_to_pf2021"],
        "site.point_source_z_max_m": strat["rise_height_consistency"]["point_source_variable_N"][
            "z_max_m"
        ],
        "site.point_source_z_neutral_m": strat["rise_height_consistency"][
            "point_source_variable_N"
        ]["z_neutral_m"],
        "site.fissure_length_for_1km_rise_m": strat["rise_height_consistency"][
            "fissure_length_required_for_observed_rise_m"
        ],
        # ---- the dataset ----
        "data.n_cores_total": ident["n_cores"],
        "data.n_cores_used": g["blocky"]["n_cores"],
        "data.n_cores_on_flow_excluded": steady["excluded"]["on_flow_cores"],
        "data.n_sieve_fractions": 4,
        "data.n_cores_in_polydisperse_fit": uns["n_cores_used"],
        "data.n_observations": 4 * uns["n_cores_used"],
        "data.n_cores_note": (
            "132 cores published; 8 logged 'on flow' are excluded, leaving 124 "
            "for the polydisperse fit.  The single-class steady benchmark fit uses 123, "
            "because one core has no positive 250-500 um measurement."
        ),
        "data.core_diameter_small_m": 0.069,
        "data.core_diameter_large_m": 0.070826,
        "data.core_area_step_percent": 5.3,
        "data.citation": "clague2009widespread",
        # ---- forward model verification ----
        "forward.mtt_point_coefficient": fwd["mtt_closure_verification"]["point_c_q_at_z_max"],
        "forward.mtt_point_published": 3.52,
        "forward.mtt_point_rel_error": fwd["mtt_closure_verification"]["point_relative_error"],
        "forward.mtt_point_at_neutral_level": fwd["mtt_closure_verification"][
            "point_c_q_at_z_neutral"
        ],
        "forward.mtt_planar_coefficient": fwd["mtt_closure_verification"]["planar_c_q_at_z_max"],
        "forward.mtt_planar_rel_error": fwd["mtt_closure_verification"]["planar_relative_error"],
        "forward.gaussian_limit_rel_error": fwd["kernel_limits"]["gaussian_limit_max_rel_error"],
        "forward.theorem1_rel_error": fwd["theorem1"]["time_reversal_max_rel_difference"],
        "forward.mass_fraction_within_L": fwd["benchmark_pf2021"]["mass_fraction_within_L"],
        "forward.unsteady_ratio_1km": fwd["unsteady_kernel"]["ratio_unsteady_to_quasi_steady"][
            "1km"
        ],
        "forward.unsteady_ratio_8km": fwd["unsteady_kernel"]["ratio_unsteady_to_quasi_steady"][
            "8km"
        ],
        "forward.w_s_blocky_250_500_ms": fwd["settling_law"]["w_s_by_shape_and_fraction_m s^-1"][
            "blocky"
        ]["250-500um"],
        "forward.w_s_sheet_250_500_ms": fwd["settling_law"]["w_s_by_shape_and_fraction_m s^-1"][
            "sheet"
        ]["250-500um"],
        "forward.shape_phi_factor": fwd["settling_law"]["shape_is_first_order"]["phi_factor"],
        "forward.settling_citation": "barreyre2011dispersal",
        # ---- identifiability ----
        "identify.modes_one_fraction": ident["H2_verdict"]["n_res_one_class"],
        "identify.modes_four_fractions": ident["H2_verdict"]["n_res_four_classes"],
        "identify.picard_truncation": ident["picard"]["truncation_index"],
        "identify.picard_condition_holds": ident["picard"]["verdict"]["condition_satisfied"],
        "identify.sigma_log_nugget": ident["noise_model"]["sigma_log_nugget"],
        "identify.front_gain_modes_waning": ident["front_constraint"]["waning_Q"]["_gain_at_1pct"],
        # ---- synthetic validation ----
        "synthetic.steady_bias_constant": syn["cases"]["constant"]["steady_fit"][
            "energy_ratio_steady_over_true"
        ],
        "synthetic.steady_bias_waning": syn["cases"]["waning"]["steady_fit"][
            "energy_ratio_steady_over_true"
        ],
        "synthetic.steady_bias_two_pulse": syn["cases"]["two_pulse"]["steady_fit"][
            "energy_ratio_steady_over_true"
        ],
        "synthetic.max_rhat": max(v["diagnostics"]["max_rhat"] for v in syn["cases"].values()),
        "sbc.n_replicates": sbc["n_replicates"],
        "sbc.mass_p_value": sbc["uniformity"]["mass_total"]["p_value"],
        "sbc.q_mean_p_value": sbc["uniformity"]["q_mass_weighted_mean"]["p_value"],
        "sbc.log_q_sd_p_value": sbc["uniformity"]["log_q_sd"]["p_value"],
        "sbc.log_q_sd_calibrated": sbc["uniformity"]["log_q_sd"]["uniform_at_5pct"],
        # ---- the NESCA result ----
        "nesca.L_fitted_m": g["blocky"]["L_m"],
        "nesca.L_z_score": g["blocky"]["L_z_score_vs_published"],
        "nesca.shape_used": uns["shape_used"],
        "nesca.mass_total_kg": p["mass_total_kg"]["median"],
        "nesca.mass_total_ci95_kg": p["mass_total_kg"]["ci95"],
        "nesca.q_mean_m3s": p["q_mass_weighted_mean_m^3s^-1"]["median"],
        "nesca.q_mean_ci95_m3s": p["q_mass_weighted_mean_m^3s^-1"]["ci95"],
        "nesca.phi_W": p["phi_at_mass_weighted_mean_W"]["median"],
        "nesca.phi_ci95_W": p["phi_at_mass_weighted_mean_W"]["ci95"],
        "nesca.log_q_sd": p["log_q_sd"]["median"],
        "nesca.sigma_log": p["sigma_log"]["median"],
        "nesca.max_rhat": uns["diagnostics"]["max_rhat"],
        "nesca.n_divergences": uns["diagnostics"]["n_divergences"],
        "nesca.phi_blocky_W": g["blocky"]["phi_W"],
        "nesca.phi_long_W": g["long"]["phi_W"],
        "nesca.phi_sheet_W": g["sheet"]["phi_W"],
        "nesca.ppc_classes_with_trend": uns["H2_single_nu_verdict"][
            "classes_with_systematic_radial_trend"
        ],
        # ---- clast shape, inferred from the deposit ----
        "shape.L_63_125_m": shape["length_scales_m"]["63-125um"],
        "shape.L_125_250_m": shape["length_scales_m"]["125-250um"],
        "shape.L_250_500_m": shape["length_scales_m"]["250-500um"],
        "shape.L_over500_m": shape["length_scales_m"][">500um"],
        "shape.extreme_ratio_observed": shape["extreme_ratio_observed"],
        "shape.extreme_ratio_ci95": shape["extreme_ratio_ci95"],
        "shape.ratio_predicted_blocky": shape["by_shape"]["blocky"]["extreme_ratio_predicted"],
        "shape.ratio_predicted_sheet": shape["by_shape"]["sheet"]["extreme_ratio_predicted"],
        "shape.chi2_blocky": shape["by_shape"]["blocky"]["chi2"],
        "shape.chi2_sheet": shape["by_shape"]["sheet"]["chi2"],
        "shape.chi2_dof": shape["by_shape"]["sheet"]["dof"],
        "shape.best_volcaniclast": shape["best_fit_volcaniclast_shape"],
        "shape.delta_chi2": shape["delta_chi2_between_best_two_volcaniclast_shapes"],
        "shape.discrimination_decisive": shape["discrimination_is_decisive"],
        # ---- assumptions used in the energy budget ----
        # Duration is not constrained by the deposit.  The reference value below
        # fixes the tabulated numbers; the verdict is reported against the sweep.
        "sources.duration_reference_s": mc["delivery_test"]["_duration_reference_s"],
        "sources.duration_sweep_min_h": mc["duration_sensitivity"]["duration_h"][0],
        "sources.duration_sweep_max_h": mc["duration_sensitivity"]["duration_h"][-1],
        "sources.lava_duration_exponent": mc["duration_sensitivity"]["_lava_power_law_exponent"],
        "sources.lava_crossing_duration_h": mc["duration_sensitivity"]["_lava_crossing_duration_h"],
        "sources.lava_area_m2": 15.0e6,
        "sources.lava_volume_m3": 4.5e7,
        "sources.observed_megaplume_rise_m": 1000.0,
        "nesca.n_cores_used": uns["n_cores_used"],
        # ---- vent position, sampled rather than profiled ----
        "vent.prior_scale_m": vs["vent_prior"]["scale_m"],
        "vent.x_median_m": vb["vent_x_m"]["median"],
        "vent.y_median_m": vb["vent_y_m"]["median"],
        "vent.x_ci95_m": vb["vent_x_m"]["ci95"],
        "vent.y_ci95_m": vb["vent_y_m"]["ci95"],
        "vent.offset_from_centroid_m": vb["vent_offset_from_centroid_m"]["median"],
        "vent.max_rhat": vb["max_rhat"],
        "vent.phi_ci95_sampled_W": vb["phi_at_mass_weighted_mean_W"]["ci95"],
        "vent.phi_ci95_profiled_W": p["phi_at_mass_weighted_mean_W"]["ci95"],
        "vent.mass_total_kg": vb["mass_total_kg"]["median"],
        # ---- heat flux marginalised over clast shape ----
        "shape.weight_blocky": vs["shape_weights"]["blocky"],
        "shape.weight_long": vs["shape_weights"]["long"],
        "shape.weight_sheet": vs["shape_weights"]["sheet"],
        "shape.phi_blocky_sampled_W": vs["by_shape"]["blocky"]["phi_at_mass_weighted_mean_W"][
            "median"
        ],
        "shape.phi_long_sampled_W": vs["by_shape"]["long"]["phi_at_mass_weighted_mean_W"]["median"],
        "shape.phi_sheet_sampled_W": vs["by_shape"]["sheet"]["phi_at_mass_weighted_mean_W"][
            "median"
        ],
        "shape.phi_marginal_W": vm["phi_W"]["median"],
        "shape.phi_marginal_ci95_W": vm["phi_W"]["ci95"],
        # ---- heat flux across the clast-shape axis ----
        "nesca.phi_poly_blocky_W": p["phi_at_mass_weighted_mean_W"]["median"],
        "nesca.phi_poly_sheet_W": (
            p["phi_at_mass_weighted_mean_W"]["median"]
            * (
                fwd["settling_law"]["w_s_by_shape_and_fraction_m s^-1"]["sheet"]["250-500um"]
                / fwd["settling_law"]["w_s_by_shape_and_fraction_m s^-1"]["blocky"]["250-500um"]
            )
            ** (4.0 / 3.0)
        ),
        "nesca.phi_ladder_note": (
            "Two Phi ladders exist and must not be mixed.  nesca.phi_blocky_W / "
            "phi_long_W / phi_sheet_W are the STEADY SINGLE-CLASS benchmark-fit values "
            "(1.27 / 0.76 / 0.45 TW).  nesca.phi_poly_* are the POLYDISPERSE "
            "posterior scaled across the shape axis (1.62 / 0.58 TW).  Figures "
            "and text must state which ladder they use."
        ),
        # ---- source discrimination ----
        "sources.megaplume_heat_min_J": dt["_megaplume_heat_range_J"][0],
        "sources.megaplume_heat_max_J": dt["_megaplume_heat_range_J"][1],
        "sources.lava_energy_J": dt["lava_cooling"]["energy_delivered_J"],
        "sources.lava_energy_over_floor": dt["lava_cooling"]["energy_over_megaplume_minimum"],
        "sources.lava_reservoir_J": dt["lava_cooling"]["reservoir_available_J"],
        "sources.volatile_energy_J": dt["volatile_exsolution"]["energy_delivered_J"],
        "sources.volatile_energy_over_floor": dt["volatile_exsolution"][
            "energy_over_megaplume_minimum"
        ],
        "sources.dyke_energy_J": dt["dyke_heating"]["energy_delivered_J"],
        "sources.dyke_energy_over_floor": dt["dyke_heating"]["energy_over_megaplume_minimum"],
        "sources.hydrothermal_energy_J": dt["hydrothermal_evacuation"]["energy_delivered_J"],
        "sources.hydrothermal_energy_over_floor": dt["hydrothermal_evacuation"][
            "energy_over_megaplume_minimum"
        ],
        "sources.log10B_lava": b10("lava_cooling"),
        "sources.log10B_volatile": b10("volatile_exsolution"),
        "sources.log10B_dyke": b10("dyke_heating"),
        "sources.log10B_hydrothermal": b10("hydrothermal_evacuation"),
        "sources.evidence_reference": best,
        "sources.dyke_prior_swing_log10": (
            max(ev["dyke_heating"][x]["logz"] for x in ev["dyke_heating"])
            - min(ev["dyke_heating"][x]["logz"] for x in ev["dyke_heating"])
        )
        / ln10,
        "sources.hydrothermal_prior_swing_log10": (
            max(ev["hydrothermal_evacuation"][x]["logz"] for x in ev["hydrothermal_evacuation"])
            - min(ev["hydrothermal_evacuation"][x]["logz"] for x in ev["hydrothermal_evacuation"])
        )
        / ln10,
    }
    m["_sources"] = picked_keys(m)
    OUT.write_text(json.dumps(m, indent=2, sort_keys=False))
    n = sum(1 for k in m if not k.startswith("_"))
    print(f"wrote {OUT} with {n} keys")
    for k in (
        "nesca.phi_W",
        "nesca.L_fitted_m",
        "sources.log10B_lava",
        "identify.modes_four_fractions",
    ):
        print(f"  {k} = {m[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
