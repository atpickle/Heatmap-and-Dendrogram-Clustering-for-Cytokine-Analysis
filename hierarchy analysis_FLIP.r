# ---- Packages ----
if (!requireNamespace("factoextra", quietly = TRUE)) install.packages("factoextra")
if (!requireNamespace("readxl", quietly = TRUE)) install.packages("readxl")
if (!requireNamespace("pheatmap", quietly = TRUE)) install.packages("pheatmap")
if (!requireNamespace("ragg", quietly = TRUE)) install.packages("ragg")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")

suppressPackageStartupMessages({
  library(factoextra)
})

# ---- Load data (CSV or XLSX) ----
file_path <- "/Users/allisonpickle/Desktop/260331_Cytokine Analysis/May-20-2026-Rat-Cytokine-27-Plex-Discovery-Assay®-Pickle-71391-Tissue-Cell-Culture-Supernatant_extracted.xlsx"  # <-- CHANGE ME
xlsx_sheet <- "260508_NIH Figuresw3"  # e.g., 1 OR "Sheet1"

# ==============================================================================
# NEW OPTION: Preserve sample order from input file?
# Set to TRUE to keep samples in left-to-right order as they appear in your file
# Set to FALSE to let hierarchical clustering determine sample order
# ==============================================================================
preserve_sample_order <- TRUE  # <-- CHANGE ME (TRUE or FALSE)

if (grepl("\\.xlsx$", file_path, ignore.case = TRUE)) {
  data_raw <- readxl::read_excel(path = file_path, sheet = xlsx_sheet)
  data_raw <- as.data.frame(data_raw, check.names = FALSE)
} else {
  data_raw <- read.csv(file_path, check.names = FALSE)
}

# Quick checks
head(data_raw)
tail(data_raw)
str(data_raw)

# ---- Expect FLIPPED layout ----
# Column 1 = feature names (cytokines), remaining columns = samples (headers in row 1)
feature_col <- names(data_raw)[1]
if (ncol(data_raw) < 2) stop("Expected at least 2 columns: feature column + >=1 sample column.")

data_raw[[feature_col]] <- as.character(data_raw[[feature_col]])

# EDIT: make feature labels plot-safe (avoid α/β conversion errors)
data_raw[[feature_col]] <- gsub("α", "alpha", data_raw[[feature_col]], fixed = TRUE)
data_raw[[feature_col]] <- gsub("β", "beta",  data_raw[[feature_col]], fixed = TRUE)

if (anyNA(data_raw[[feature_col]]) || any(data_raw[[feature_col]] == "")) {
  stop("First column (feature names) contains missing/blank values.")
}

# Use first column as rownames (feature names), then drop it
feature_names <- make.unique(data_raw[[feature_col]])
rownames(data_raw) <- feature_names
mat <- data_raw[, setdiff(names(data_raw), feature_col), drop = FALSE]

# STORE ORIGINAL SAMPLE ORDER from input file
original_sample_order <- colnames(mat)

# Coerce all sample columns to numeric safely (Excel often imports as character)
mat_num <- as.data.frame(lapply(mat, function(x) suppressWarnings(as.numeric(x))), check.names = FALSE)

# EDIT: carry over Column A labels as rownames for cytokines/features
rownames(mat_num) <- feature_names

# Drop features that are all NA after coercion
keep_feature <- rowSums(!is.na(mat_num)) > 0
mat_num <- mat_num[keep_feature, , drop = FALSE]

# ---- Build a sample-by-feature data frame for sample clustering ----
# df: rows = samples, cols = features
df <- as.data.frame(t(as.matrix(mat_num)), check.names = FALSE)
rownames(df) <- make.unique(as.character(rownames(df)))

# Keep only numeric columns (extra safety)
numeric_cols <- vapply(df, is.numeric, logical(1))
df <- df[, numeric_cols, drop = FALSE]

# Remove rows with any NA (same behavior as your original script)
df <- na.omit(df)

# ---- Output folder for exported plots ----
sheet_tag <- if (grepl("\\.xlsx$", file_path, ignore.case = TRUE)) {
  gsub("[^A-Za-z0-9]+", "_", as.character(xlsx_sheet))
} else {
  "csv"
}

out_dir <- file.path(dirname(file_path), paste0("hc_outputs_", sheet_tag))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

base_tag <- paste0(tools::file_path_sans_ext(basename(file_path)), "__", sheet_tag, "__FLIP")


# =============================================================================
# (A) SAMPLE CLUSTERING (based on cytokine results)
# =============================================================================

# Scale features (z-score per cytokine column) then compute distances across samples
df.scaled <- scale(df)
res.dist.sample <- dist(x = df.scaled, method = "euclidean")
res.hc.sample <- hclust(d = res.dist.sample, method = "complete")

# EDIT: cap k_sample to allowable range (prevents cutree() errors)
k_sample <- 4
k_sample <- min(k_sample, nrow(df))
if (k_sample < 2) stop("Need at least 2 samples for k-sample clustering plot.")

# --- Export SAMPLE k-dendrogram (factoextra) ---
ragg::agg_png(file.path(out_dir, paste0(base_tag, sprintf("__SAMPLE_dendrogram_k%d.png", k_sample))),
              width = 2400, height = 2200, res = 200)
print(
  factoextra::fviz_dend(
    res.hc.sample,
    cex = 0.8,
    lwd = 0.5,
    k = k_sample,
    k_colors = c("red", "green3", "blue", "magenta"),
    rect = FALSE
  ) +
    ggplot2::coord_cartesian(clip = "off") +                  # EDIT: prevents label clipping
    ggplot2::scale_x_continuous(breaks = NULL) +              # EDIT: remove numeric x ticks/labels
    ggplot2::theme(
      plot.margin = ggplot2::margin(t = 10, r = 10, b = 260, l = 10),
      axis.ticks.x = ggplot2::element_blank(),
      axis.text.x  = ggplot2::element_text(angle = 90, vjust = 0.5, hjust = 1),
      axis.title.x = ggplot2::element_blank(),
      axis.line.x  = ggplot2::element_blank()
    )
)
dev.off()

pdf(file.path(out_dir, paste0(base_tag, sprintf("__SAMPLE_dendrogram_k%d.pdf", k_sample))),
    width = 12, height = 11)
print(
  factoextra::fviz_dend(
    res.hc.sample,
    cex = 0.8,
    lwd = 0.5,
    k = k_sample,
    k_colors = c("red", "green3", "blue", "magenta"),
    rect = FALSE
  ) +
    ggplot2::coord_cartesian(clip = "off") +                  # EDIT
    ggplot2::scale_x_continuous(breaks = NULL) +              # EDIT
    ggplot2::theme(
      plot.margin = ggplot2::margin(t = 10, r = 10, b = 260, l = 10),
      axis.ticks.x = ggplot2::element_blank(),
      axis.text.x  = ggplot2::element_text(angle = 90, vjust = 0.5, hjust = 1),
      axis.title.x = ggplot2::element_blank(),
      axis.line.x  = ggplot2::element_blank()
    )
)
dev.off()

message("Sample dendrogram plots exported to: ", out_dir)

# =============================================================================
# (B) FEATURE/CYTOKINE CLUSTERING (based on Column A items)
# =============================================================================

# Scale *rows* so features are comparable across samples
feat_scaled <- t(scale(t(as.matrix(mat_num))))
feat_dist <- dist(feat_scaled, method = "euclidean")
res.hc.feature <- hclust(feat_dist, method = "complete")

# --- Export feature dendrograms (base plot) ---
ragg::agg_png(file.path(out_dir, paste0(base_tag, "__FEATURE_dendrogram_base.png")),
              width = 2200, height = 1400, res = 200)
par(mar = c(10, 4, 4, 2) + 0.1)
plot(res.hc.feature, main = "Feature clustering (cytokines, complete linkage)", xlab = "", sub = "")
dev.off()

pdf(file.path(out_dir, paste0(base_tag, "__FEATURE_dendrogram_base.pdf")),
    width = 11, height = 8.5)
par(mar = c(10, 4, 4, 2) + 0.1)
plot(res.hc.feature, main = "Feature clustering (cytokines, complete linkage)", xlab = "", sub = "")
dev.off()

# EDIT: cap k_feature to allowable range
k_feature <- 4
k_feature <- min(k_feature, nrow(mat_num))
if (k_feature < 2) stop("Need at least 2 features (rows) for k-feature clustering plot.")

# --- Export FEATURE k-dendrogram (factoextra) ---
ragg::agg_png(file.path(out_dir, paste0(base_tag, sprintf("__FEATURE_dendrogram_k%d.png", k_feature))),
              width = 2400, height = 2000, res = 200)
print(
  factoextra::fviz_dend(
    res.hc.feature,
    cex = 0.6,
    lwd = 0.7,
    k = k_feature,
    k_colors = c("red", "green3", "blue", "magenta"),
    rect = FALSE
  ) +
    ggplot2::coord_cartesian(clip = "off") +                  # EDIT: prevents label clipping
    ggplot2::scale_x_continuous(breaks = NULL) +              # EDIT: remove numeric x ticks/labels
    ggplot2::theme(
      plot.margin = ggplot2::margin(t = 10, r = 10, b = 220, l = 10),
      axis.ticks.x = ggplot2::element_blank(),
      axis.title.x = ggplot2::element_blank(),
      axis.line.x  = ggplot2::element_blank()
    )
)
dev.off()

pdf(file.path(out_dir, paste0(base_tag, sprintf("__FEATURE_dendrogram_k%d.pdf", k_feature))),
    width = 12, height = 10)
print(
  factoextra::fviz_dend(
    res.hc.feature,
    cex = 0.6,
    lwd = 0.7,
    k = k_feature,
    k_colors = c("red", "green3", "blue", "magenta"),
    rect = FALSE
  ) +
    ggplot2::coord_cartesian(clip = "off") +                  # EDIT
    ggplot2::scale_x_continuous(breaks = NULL) +              # EDIT
    ggplot2::theme(
      plot.margin = ggplot2::margin(t = 10, r = 10, b = 220, l = 10),
      axis.ticks.x = ggplot2::element_blank(),
      axis.title.x = ggplot2::element_blank(),
      axis.line.x  = ggplot2::element_blank()
    )
)
dev.off()

message("Feature (Column A) dendrogram plots exported to: ", out_dir)

# =============================================================================
# Heatmap vs a reference sample: cytokines on Y, samples on X
# =============================================================================

ref_sample <- "Control (6hr)"  # <-- CHANGE ME
if (!(ref_sample %in% colnames(mat_num))) {
  stop(paste0(
    "Reference sample '", ref_sample, "' not found among sample columns. ",
    "Available columns include: ", paste(head(colnames(mat_num), 10), collapse = ", "), " ..."
  ))
}

pseudocount <- 1e-9
ref_vals_feat <- as.numeric(mat_num[[ref_sample]])

fc_heat <- log2((as.matrix(mat_num) + pseudocount) / (ref_vals_feat + pseudocount))
fc_heat[, ref_sample] <- 0

# Determine sample order based on user preference
if (preserve_sample_order) {
  # Keep samples in their original left-to-right order from input file
  sample_order <- intersect(original_sample_order, colnames(fc_heat))
  fc_heat <- fc_heat[, sample_order, drop = FALSE]
  cluster_cols_setting <- FALSE
  order_tag <- "_ORIGINAL_ORDER"
  message("Using ORIGINAL sample order from input file (left to right)")
} else {
  # Use hierarchical clustering to determine sample order
  cluster_cols_setting <- res.hc.sample
  order_tag <- "_CLUSTERED"
  message("Using HIERARCHICAL CLUSTERING to determine sample order")
}

heatmap_png <- file.path(out_dir, paste0(base_tag, "__heatmap_log2FC_vs_", ref_sample, "__samples_on_x", order_tag, ".png"))
heatmap_pdf <- file.path(out_dir, paste0(base_tag, "__heatmap_log2FC_vs_", ref_sample, "__samples_on_x", order_tag, ".pdf"))

lim <- max(abs(fc_heat), na.rm = TRUE)

# PNG via ragg (Unicode-safe)
ragg::agg_png(heatmap_png, width = 2600, height = 1700, res = 200)
pheatmap::pheatmap(
  mat = fc_heat,
  color = grDevices::colorRampPalette(c("blue", "white", "red"))(101),
  breaks = seq(-lim, lim, length.out = 102),
  cluster_rows = res.hc.feature,
  cluster_cols = cluster_cols_setting,
  gaps_row = NULL,
  gaps_col = NULL,
  fontsize_row = 7,
  fontsize_col = 9,
  angle_col = 45,
  main = paste0("log2 fold-change vs ", ref_sample, " (samples on X axis)")
)
dev.off()

# PDF output
pdf(heatmap_pdf, width = 12, height = 8.5)
pheatmap::pheatmap(
  mat = fc_heat,
  color = grDevices::colorRampPalette(c("blue", "white", "red"))(101),
  breaks = seq(-lim, lim, length.out = 102),
  cluster_rows = res.hc.feature,
  cluster_cols = cluster_cols_setting,
  gaps_row = NULL,
  gaps_col = NULL,
  fontsize_row = 7,
  fontsize_col = 9,
  angle_col = 45,
  main = paste0("log2 fold-change vs ", ref_sample, " (samples on X axis)")
)
dev.off()

message("Reference heatmap exported to:\n  ", heatmap_png, "\n  ", heatmap_pdf)