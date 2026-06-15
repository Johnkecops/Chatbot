#!/usr/bin/env Rscript
# SVM Chatbot Pipeline
# Run this script to execute all steps in order.
# Working directory must be set to "SVM Chatbot File/" before running.
#
# Usage: Rscript run_pipeline.R
#
# Steps:
#   Database.r              — build reference data frame, export to output.xlsx
#   Divide and training data.r — 80/20 train/test split (set.seed=123)
#   Step 1.r                — text preprocessing, DocumentTermMatrix
#   Step 2.r                — assemble training matrix
#   Step 3.r                — train LinearSVM (e1071, kernel="linear", cost=100)
#   Step 4.r                — defines pred() for interactive inference
#
# After sourcing, call pred("your question") in the R console.

setwd(dirname(sys.frame(1)$ofile))   # ensure CWD = this file's directory

source("Database.r")
source("Divide and training data.r")
source("Step 1.r")
source("Step 2.r")
source("Step 3.r")
source("Step 4.r")

message("Pipeline complete. Call pred(\"your question\") to query the model.")
