rm(list = ls())
read_mturk_res = function(data_path){
  # dirs = list.dirs(data_path, full.names = TRUE, recursive = FALSE)
  files <- list.files(path = data_path, pattern = "\\.csv$", full.names = TRUE)
  require(ramify)
  output = list()
  index = 1
  for (f in files){
    df = read.csv(f)
    df$repay = as.numeric(gsub("\\[|\\]", "", df$repay))
    # df$adv.action = as.numeric(gsub("\\[|\\]", "",df$adv.action))
    df$id = f
    df$trial = c(1:10)
    output[[index]] = df
    index = index + 1
  }
  require(data.table)
  all_s = as.data.table(as.data.frame(rbindlist(output)))
  all_s
}

mturk_d = read_mturk_res("sim_data/gpt/gpt-3.5-turbo")
mturk_sumr = mturk_d[, c("id", "repay", "investment")]
mturk_sumr$action = mturk_sumr$investment
mturk_sumr$reward = floor(mturk_sumr$repay)
mturk_sumr$repay = NULL
mturk_sumr$investment = NULL

mturk_d1 = read_mturk_res("sim_data/gpt/gpt-3.5-turbo1")
mturk_sumr1 = mturk_d1[, c("id", "repay", "investment")]
mturk_sumr1$action = mturk_sumr1$investment
mturk_sumr1$reward = floor(mturk_sumr1$repay)
mturk_sumr1$repay = NULL
mturk_sumr1$investment = NULL


mturk = rbind(mturk_sumr, mturk_sumr1)
write.csv(mturk, "sim_data/gpt/gpt-3.5-turbo.csv", row.names = FALSE)

mturk_d = read_mturk_res("sim_data/gpt/gpt-4-turbo")
mturk_sumr = mturk_d[, c("id", "repay", "investment")]
mturk_sumr$action = mturk_sumr$investment
mturk_sumr$reward = floor(mturk_sumr$repay)
mturk_sumr$repay = NULL
mturk_sumr$investment = NULL

write.csv(mturk_sumr, "sim_data/gpt/gpt-4-turbo.csv", row.names = FALSE)


mturk_d = read_mturk_res("sim_data/gemini/gemini_1.5")
mturk_sumr = mturk_d[, c("id", "repay", "investment")]
mturk_sumr$action = mturk_sumr$investment
mturk_sumr$reward = floor(mturk_sumr$repay)
mturk_sumr$repay = NULL
mturk_sumr$investment = NULL

write.csv(mturk_sumr, "sim_data/gemini/gemini_1.5.csv", row.names = FALSE)
