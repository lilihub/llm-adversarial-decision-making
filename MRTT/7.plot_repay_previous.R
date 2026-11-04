library(plyr)
library(ggplot2)

rm(list = ls())
read_mturk_res = function(data_path){
  # dirs = list.dirs(data_path, full.names = TRUE, recursive = FALSE)
  dirs = list.files(data_path, full.names = TRUE, recursive = FALSE)
  require(ramify)
  output = list()
  index = 1
  for (d in dirs){
    f = paste0(d, '/data/output.csv')
    df = read.csv(f)
    df$repay = as.numeric(gsub("\\[|\\]", "", df$repay))
    df$adv.action = as.numeric(gsub("\\[|\\]", "",df$adv.action))
    df$id = d
    df$trial = c(1:10)
    df$trustee_earn =  3 * df$investment - df$repay
    df$repay_precent =  100 * df$repay / (df$investment * 3 + 0.0000001)
    df$learner.discr = floor(clip(df$investment - 0.0001, 0, 1000) / 4)
    df$learner.discr = floor(pmax(0, pmin(df$investment - 0.0001, 1000)) / 4)
    df$investor_earn = (20 - df$investment) + df$repay
    output[[index]] = df
    index = index + 1
  }
  require(data.table)
  all_s = as.data.table(as.data.frame(rbindlist(output)))
  all_s[, prev_repay_precent := shift(repay_precent, 1), by=.(id)]
  all_s$prev_repay_precent_disc = cut(all_s$prev_repay_precent, 5, method="length", na.omit=FALSE)
  all_s$investement.discr = floor(clip(all_s$investment - 0.0001, 0, 1000) / 4) + 1
  all_s
}
read_sim_gpt = function(data_path){
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
    df$trustee_earn =  3 * df$investment - df$repay
    df$repay_precent =  100 * df$repay / (df$investment * 3 + 0.0000001)
    df$learner.discr = floor(clip(df$investment - 0.0001, 0, 1000) / 4)
    df$learner.discr = floor(pmax(0, pmin(df$investment - 0.0001, 1000)) / 4)
    df$investor_earn = (20 - df$investment) + df$repay
    output[[index]] = df
    index = index + 1
  }
  require(data.table)
  all_s = as.data.table(as.data.frame(rbindlist(output)))
  all_s[, prev_repay_precent := shift(repay_precent, 1), by=.(id)]
  break_points = c(-0.1, 20, 40, 60, 80, 100)
  all_s$prev_repay_precent_disc = cut(all_s$prev_repay_precent, breaks=break_points, method="length", na.omit=FALSE)
  all_s$investement.discr = floor(clip(all_s$investment - 0.0001, 0, 1000) / 4) + 1
  all_s
}

plot_prev_repay = function(data){
  require(plyr)
  require(ggplot2)
  library("RColorBrewer")
  cc = brewer.pal(n = 8, name = "Dark2")
  dark2_colors <- c("#1b9e77", "#7570b3", "#e7298a", "#d95f02", "#e6ab02")
  data = ddply(data, c("id", "prev_repay_precent_disc", "group"), function(x){data.frame(investment=mean(x$investment))})
  ggplot(subset(data, !(data$prev_repay_precent_disc == "NA")), aes(y = investment, x =prev_repay_precent_disc, fill=group)) +
    geom_bar(stat = "summary", fun.y = "mean", position = "dodge") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("investment in current trial") +
    xlab("%repayment in previous trial") +
    scale_x_discrete(labels= c("0-20", "20-40", "40-60", "60-80", "80-100"))  +
    theme_bw() +
    scale_fill_manual(values=dark2_colors,
                      labels = c("Human", "GPT-3.5", "GPT-4", "Gemini-1.5", "DeepSeek-V3")) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
    theme(axis.title.y = element_text(size = 18), # Increase Y-axis label size and make it bold
          axis.title.x = element_text(size = 18),
          axis.text.x = element_text(size = 14), # Increase X-axis tick text size
          axis.text.y = element_text(size = 14),
          legend.text = element_text(size = 14))
  
}
subj_rnd = read_mturk_res('human_data/RND/')
subj_rnd$group = "Human vs RND"
subj_rnd$cat = "RND"

gpt_rnd = read_sim_gpt("sim_data/gpt/gpt-3.5-turbo")
gpt_rnd$group = "GPT-3.5 vs RND"
gpt_rnd$cat = "RND"

gpt4_rnd = read_sim_gpt("sim_data/gpt/gpt-4-turbo")
gpt4_rnd$group = "GPT-4 vs RND"
gpt4_rnd$cat = "RND"

gemini_rnd = read_sim_gpt("sim_data/gemini/gemini_1.5")
gemini_rnd$group = "gemini-1.5 vs RND"
gemini_rnd$cat = "RND"

deepseek_rnd = read_sim_gpt("sim_data/deepseek/deepseek-chat")
deepseek_rnd$group = "DeepSeek-V3 vs RND"
deepseek_rnd$cat = "RND"

alld_rnd = rbind(subj_rnd[,c("id", "prev_repay_precent_disc", "investment", "group", "cat")],
                 gpt_rnd[,c("id", "prev_repay_precent_disc", "investment", "group", "cat")])
alld_rnd = rbind(alld_rnd,
                 gpt4_rnd[,c("id", "prev_repay_precent_disc", "investment", "group", "cat")])
alld_rnd = rbind(alld_rnd,
                 gemini_rnd[,c("id", "prev_repay_precent_disc", "investment", "group", "cat")])
alld_rnd = rbind(alld_rnd,
                 deepseek_rnd[,c("id", "prev_repay_precent_disc", "investment", "group", "cat")])


alld_rnd$group <- factor(alld_rnd$group, levels = c("Human vs RND", "GPT-3.5 vs RND", 
                                                    "GPT-4 vs RND", "gemini-1.5 vs RND",
                                                    "DeepSeek-V3 vs RND"))
plot_prev_repay(alld_rnd)
ggsave("plots/previous_repay.pdf", width=16, height=18, unit="cm", useDingbats=FALSE)



