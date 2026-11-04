
############# Merge all the experimental csv files to one #############
paper_mode = TRUE
source('R/utils/helper.R')
source("R/utils/summary_utils.R")

# Set the path to the folder containing the CSV files
# For GPT-3.5
# folder_path <- "sim_data/gpt/gpt-3.5-turbo"
# For GPT-4
# folder_path <- "sim_data/gpt/gpt-4-turbo"
# For Gemini
# folder_path <- "sim_data/gemini/gemini_1.5"
# For DeepSeek
folder_path <- "sim_data/deepseek/deepseek-chat"

# List all CSV files in the folder
csv_files <- list.files(path = folder_path, pattern = ".csv", full.names = TRUE)
data_frames <- list()
# Loop through the list of CSV files and read each file
for (file in csv_files) {
  data <- read.csv(file)
  data_frames <- append(data_frames, list(data))
}
combined_data <- do.call(rbind, data_frames)
colnames(combined_data)[2] <- 'id' 
# write.csv(combined_data, file = "sim_data/gpt/gpt-3.5-turbo.csv", row.names = FALSE)
# write.csv(combined_data, file = "sim_data/gpt/gpt-4-turbo.csv", row.names = FALSE)
# write.csv(combined_data, file = "sim_data/gemini/gemini_1.5.csv", row.names = FALSE)
write.csv(combined_data, file = "sim_data/deepseek/deepseek-chat.csv", row.names = FALSE)






