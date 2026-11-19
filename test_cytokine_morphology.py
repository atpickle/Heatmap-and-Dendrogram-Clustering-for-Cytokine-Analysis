import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from Cytokine_Morphology_Analysis import Plot_Cytokine_Morphology_Plots, cytokine_source_suffix
from Cytokine_Scatter_Analysis import Plot_Cytokine_Scatter_Plots


class PDMorphologyTests(unittest.TestCase):
    def test_pd_matching_aliases_and_well_summaries(self):
        cytokines = pd.DataFrame({
            "Sample": ["6hr Ctrl", "2mM PD", "24hr 100ng",
                       "24hrPD_100ng", "24hrPD_100ng24hr", "1hrPD_100ng", "1hrPD_5ng"],
            "Rep": ["Rep 1"] * 7,
            "Cytokine": ["IL-6"] * 7,
            "Obs_Conc": [1, 2, 3, 4, 6, 7, 99],
        })
        cytokines.attrs["source_selections"] = [{
            "sheet_name": "261018 Updated Analysis", "data_start_row": 194, "data_end_row": 235,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            metric = folder / "MM_Outputs_PD_Morphology_2mM_Area (µm²)"
            metric.mkdir()
            rows = []
            for treatment in ["Tri_Ctrl", "PD_0LPS", "100ng_LPS",
                              "24hrPD_100ng", "1hrPD_100ng", "CoPD_100ng", "1hrPD_5ng"]:
                for well, value in [(1, 2), (2, 4)]:
                    rows.append(dict(Treatment=treatment, Biological_Rep=1, Well=well,
                                     Results=value, Source="LPS_only" if treatment == "100ng_LPS" else "PD"))
            pd.DataFrame(rows).to_csv(metric / "Combined_Analysis_Data.csv", index=False)
            cyto, morphology, paths = Plot_Cytokine_Morphology_Plots(
                cytokines, folder, pd_concentration="2mM", save_results=True,
            )
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(path).stat().st_size > 10000 for path in paths))
            self.assertTrue(all(
                Path(path).parent.name ==
                "PD_Cytokine_Morphology_Plots__261018_Updated_Analysis__rows194-235"
                for path in paths
            ))
            alias = cyto.loc[cyto.Treatment.eq("24hrPD_100ng")].iloc[0]
            self.assertEqual(alias.Obs_Conc, 5)
            self.assertEqual(alias.TechnicalRows, 2)
            self.assertFalse(morphology.Treatment.str.contains("5ng|CoPD").any())
            raw = morphology.loc[morphology.Morphology_Scale.eq("raw")]
            self.assertTrue(raw.Results.eq(3).all())
            self.assertTrue(raw.SEM.eq(1).all())
            log = morphology.loc[morphology.Morphology_Scale.eq("ln")]
            np.testing.assert_allclose(log.Results, np.log([2, 4]).mean())
            np.testing.assert_allclose(log.SEM, np.log(2) / 2)

    def test_5mm_cannot_reuse_2mm_cytokines(self):
        with self.assertRaisesRegex(ValueError, "2 mM only"):
            Plot_Cytokine_Morphology_Plots(None, ".", pd_concentration="5mM", save_results=False)

    def test_legacy_timecourse_remains_available(self):
        data = pd.DataFrame({
            "Sample": ["24hr 100ng"], "Rep": ["1"], "Cytokine": ["IL-6"],
            "Obs_Conc": [2], "Time_hr": [24],
        })
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({
                "Treatment": ["24hr_100ng", "24hr_100ng"],
                "Biological_Rep": [1, 1], "Well": [1, 2], "Results": [2, 4],
            }).to_csv(Path(tmp) / "Area.csv", index=False)
            _, morphology, paths = Plot_Cytokine_Morphology_Plots(
                data, tmp, save_results=False,
            )
            self.assertEqual(set(morphology.Morphology_Scale), {"raw", "ln"})
            self.assertEqual(paths, [])

    def test_source_suffix_preserves_multiple_disjoint_selections(self):
        data = pd.DataFrame({"Sheet": ["Data"], "Excel_Row": [198]})
        data.attrs["source_selections"] = [
            {"sheet_name": "Data Set", "data_start_row": 194, "data_end_row": 235},
            {"sheet_name": "Data Set", "data_start_row": 250, "data_end_row": 260},
        ]
        self.assertEqual(
            cytokine_source_suffix(data),
            "__Data_Set__rows194-235__Data_Set__rows250-260",
        )
        data.attrs.clear()
        self.assertEqual(cytokine_source_suffix(data), "__Data__rows198-198")
        self.assertEqual(cytokine_source_suffix(pd.DataFrame()), "")

    def test_timecourse_default_folder_and_explicit_override(self):
        data = pd.DataFrame({
            "Sample": ["24hr 100ng"], "Rep": ["1"], "Cytokine": ["IL-6"],
            "Obs_Conc": [2], "Time_hr": [24],
        })
        data.attrs["source_selections"] = [{
            "sheet_name": "261018 Updated Analysis", "data_start_row": 194, "data_end_row": 235,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pd.DataFrame({
                "Treatment": ["24hr_100ng", "24hr_100ng"],
                "Biological_Rep": [1, 1], "Well": [1, 2], "Results": [2, 4],
            }).to_csv(folder / "Area.csv", index=False)
            for output_dir in (None, folder / "custom"):
                with self.subTest(output_dir=output_dir):
                    _, _, paths = Plot_Cytokine_Morphology_Plots(
                        data, folder, output_dir=output_dir, save_results=True,
                    )
                    expected = output_dir or folder / (
                        "Cytokine_Morphology_Plots__261018_Updated_Analysis__rows194-235"
                    )
                    self.assertTrue(paths)
                    self.assertTrue(all(Path(path).parent == expected for path in paths))

    def test_scatter_preserves_original_bounds_after_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.xlsx"
            raw = pd.DataFrame([[None] * 7 for _ in range(4)])
            raw.iat[0, 5] = "IL-6"
            for row, sample in enumerate(["6hr 5ng", "24hr 5ng", "48hr 5ng"], start=1):
                raw.iat[row, 0], raw.iat[row, 1], raw.iat[row, 6] = "Rep 1", sample, row
            raw.to_excel(source, sheet_name="Data Set", header=False, index=False)
            for multi_sheet in (True, False):
                with self.subTest(multi_sheet=multi_sheet):
                    kwargs = {"sheet_selections": [{"sheet_name": "Data Set"}]} if multi_sheet else {"sheet_name": "Data Set"}
                    parsed, _, _ = Plot_Cytokine_Scatter_Plots(
                        source, header_row=1, samples=["24hr 5ng"], include_48hr=False,
                        run_statistics=False, save_results=False, display_results=False, **kwargs,
                    )
                    self.assertEqual(set(parsed.Sample), {"24hr 5ng"})
                    self.assertEqual(parsed.attrs["source_selections"], [{
                        "sheet_name": "Data Set", "data_start_row": 2, "data_end_row": 4,
                    }])
                    self.assertEqual(cytokine_source_suffix(parsed), "__Data_Set__rows2-4")


if __name__ == "__main__":
    unittest.main()
