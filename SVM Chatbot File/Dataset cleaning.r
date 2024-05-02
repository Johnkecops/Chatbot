# Assuming the OCR text represents the data to be written to an Excel file
data <- data.frame(
  Symptom = c("Chest discomfort", "Faint", "Discomfort in shoulder, neck, back and jaw", "Shortness of breath", "Fractured bone", "Persistent headache", "Foreign material stuck", "Skin infection", "Protein with vision", "Fever", "Diarrhea", "Cold sweat", "Back pain", "Weakness", "Trouble in speaking, seeing, walking", "Vomit", "Stomachache"),
  Disease = c("Hypertension", "Diabetes", "Heart disease", "Liver disease", "Stroke", "Cuts", "Tumor/cancer", "Respiration disease", "Gastric", "", "", "", "", "", "", "", ""),
  Code = c("1-SpA", "2-SpB", "3-SpC", "4-SpD", "SpE", "SpF", "SpG", "SpH", "SpI", "SpJ", "SpK", "SpL", "SpM", "SpN", "SpO", "SpP", "")
)

# Install the openxlsx package if not already installed
if (!require(openxlsx)) install.packages("openxlsx")

# Load the openxlsx library
library(openxlsx)

# Write the data frame to an Excel file
write.xlsx(data, file = "output.xlsx")
