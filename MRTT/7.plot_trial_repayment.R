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
    df$repay_percent =  100 * df$repay / (df$investment * 3 + 0.0000001)
    df$learner.discr = floor(clip(df$investment - 0.0001, 0, 1000) / 4)
    df$learner.discr = floor(pmax(0, pmin(df$investment - 0.0001, 1000)) / 4)
    df$investor_earn = (20 - df$investment) + df$repay
    output[[index]] = df
    index = index + 1
  }
  require(data.table)
  all_s = as.data.table(as.data.frame(rbindlist(output)))
  all_s[, prev_repay_precent := shift(repay_percent, 1), by=.(id)]
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
    df$id = f
    df$trial = c(1:10)
    df$trustee_earn =  3 * df$investment - df$repay
    df$repay_percent =  100 * df$repay / (df$investment * 3 + 0.0000001)
    df$learner.discr = floor(clip(df$investment - 0.0001, 0, 1000) / 4)
    df$learner.discr = floor(pmax(0, pmin(df$investment - 0.0001, 1000)) / 4)
    df$investor_earn = (20 - df$investment) + df$repay
    output[[index]] = df
    index = index + 1
  }
  require(data.table)
  all_s = as.data.table(as.data.frame(rbindlist(output)))
  all_s[, prev_repay_percent := shift(repay_percent, 1), by=.(id)]
  break_points = c(-0.1, 20, 40, 60, 80, 100)
  all_s$prev_repay_percent_disc = cut(all_s$prev_repay_percent, breaks=break_points, method="length", na.omit=FALSE)
  all_s$investement.discr = floor(clip(all_s$investment - 0.0001, 0, 1000) / 4) + 1
  all_s
}

read_sim_data_real = function(data_path, sim_count, adv_action_count=5){
  dd = list()
  indx = 1
  for (i in c(0:sim_count)){
    # for human
    data = read.csv(paste0(data_path, '/events_', i,".csv"))
    data$sbj = i
    data$trial = (data$X + 1 )
    # data$percent_return = (as.integer(gsub("\\[|\\]", "", data$adv.action)) * (100 / (adv_action_count - 1)))
    data$repay = as.integer(gsub("\\[|\\]", "", data$learner.reward))
    data$investment = as.integer(gsub("\\[|\\]", "", data$real_model.action.cont))
    data$learner.discr = NA
    data$learner.discr[data$learner.action == '[[[0 0 0 0 1]]]'] = 5
    data$learner.discr[data$learner.action == '[[[0 0 0 1 0]]]'] = 4
    data$learner.discr[data$learner.action == '[[[0 0 1 0 0]]]'] = 3
    data$learner.discr[data$learner.action == '[[[0 1 0 0 0]]]'] = 2
    data$learner.discr[data$learner.action == '[[[1 0 0 0 0]]]'] = 1
    dd[[indx]] = data
    indx = indx + 1
  }
  
  require(data.table)
  all_d = as.data.frame(rbindlist(dd))
  # all_d$repay_precent = all_d$percent_return
  # all_d$investment = as.integer(gsub("\\[|\\]", "", all_d$real_model.action.cont))
  all_d$repay_percent =  100 * all_d$repay / (all_d$investment * 3 + 0.0000001)
  all_d$id = all_d$sbj
  all_d = as.data.table(all_d)
  all_d[, prev_repay_percent := shift(repay_percent, 1), by=.(id)]
  all_d$prev_repay_percent_disc = cut(all_d$prev_repay_percent, 5, method="length", na.omit=FALSE)
  all_d$investement.discr = floor(clip(all_d$investment - 0.0001, 0, 1000) / 4) + 1
  all_d$trustee_earn =  3 * all_d$investment - all_d$repay
  all_d$investor_earn = (20 - all_d$investment) + all_d$repay
  all_d = as.data.frame(all_d)
}

read_sim_data_ler = function(data_path, sim_count, adv_action_count=5){
  dd = list()
  indx = 1
  for (i in c(0:sim_count)){
    
    # for human
    data = read.csv(paste0(data_path, '/events_', i,".csv"))
    data$sbj = i
    data$trial = (data$X + 1 )
    # data$percent_return = (as.integer(gsub("\\[|\\]", "", data$adv.action)) * (100 / (adv_action_count - 1)))
    data$repay = as.integer(gsub("\\[|\\]", "", data$learner.reward))
    data$investment = as.integer(gsub("\\[|\\]", "", data$learner.action.cont))
    data$learner.discr = NA
    data$learner.discr[data$learner.action == '[[[0 0 0 0 1]]]'] = 5
    data$learner.discr[data$learner.action == '[[[0 0 0 1 0]]]'] = 4
    data$learner.discr[data$learner.action == '[[[0 0 1 0 0]]]'] = 3
    data$learner.discr[data$learner.action == '[[[0 1 0 0 0]]]'] = 2
    data$learner.discr[data$learner.action == '[[[1 0 0 0 0]]]'] = 1
    dd[[indx]] = data
    indx = indx + 1
  }
  
  require(data.table)
  all_d = as.data.frame(rbindlist(dd))
  all_d$repay_percent =  100 * all_d$repay / (all_d$investment * 3 + 0.0000001)
  all_d$id = all_d$sbj
  all_d = as.data.table(all_d)
  all_d[, prev_repay_percent := shift(repay_percent, 1), by=.(id)]
  all_d$prev_repay_percent_disc = cut(all_d$prev_repay_percent, 5, method="length", na.omit=FALSE)
  all_d$investement.discr = floor(clip(all_d$investment - 0.0001, 0, 1000) / 4) + 1
  all_d$trustee_earn =  3 * all_d$investment - all_d$repay
  all_d$investor_earn = (20 - all_d$investment) + all_d$repay
  all_d = as.data.frame(all_d)
}

plot_repay_trial = function(data){
  data_r = ddply(subset(data, investment != 0), c("trial", "id", "cat"), function(x){data.frame(repay=mean(x$repay_percent))})
  data_r$cat <- factor(data_r$cat, levels = c("Human", "GPT-3.5", "GPT-4", "Gemini-1.5", "DeepSeek-V3"))
  dark2_colors <- c("#1b9e77", "#7570b3", "#e7298a", "#d95f02", "#e6ab02")
  ggplot(subset(data_r, T), aes(y = repay, x =trial, fill=cat, group=cat)) +
    #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean",position = "dodge") +
    # stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
    #              position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("%repayment") +
    xlab("trial") +
    theme_bw() +
    scale_fill_manual(values=dark2_colors) +   # Purple from Dark2
    scale_x_discrete(limits = seq(1,10,1)) +
    coord_cartesian(ylim=c(0, 100))+
    theme(axis.title.y = element_text(size = 20), # Increase Y-axis label size and make it bold
          axis.text.x = element_text(size = 16), # Increase X-axis tick text size
          axis.text.y = element_text(size = 16),
          axis.title.x = element_text(size = 20),
          legend.position = "none")
    # guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))
}

plot_investment_trial = function(data){
  data_i = ddply(data, c("trial", "id", "cat"), function(x){data.frame(investment=mean(x$investment))})
  data_i$cat <- factor(data_i$cat, levels = c("Human", "GPT-3.5", "GPT-4", "Gemini-1.5", "DeepSeek-V3"))
  dark2_colors <- c("#1b9e77", "#7570b3", "#e7298a", "#d95f02", "#e6ab02")
  ggplot(subset(data_i, T), aes(y = investment, x =trial, fill=cat)) +
    #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean",position = "dodge") +
    # stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
    #              position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("investment") +
    xlab("trial") +
    theme_bw() +
    scale_fill_manual(values=dark2_colors) +   # Purple from Dark2
    scale_x_discrete(limits = seq(1,10,1)) +
    coord_cartesian(ylim=c(0, 20))+
    theme(text = element_text(size=8))+
    theme(axis.title.y = element_text(size = 20), # Increase Y-axis label size and make it bold
          axis.text.x = element_text(size = 16), # Increase X-axis tick text size
          axis.text.y = element_text(size = 16),
          axis.title.x = element_text(size = 20),
          legend.text = element_text(size = 16),
          legend.title = element_blank())+
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))
}

subj_max = read_mturk_res('human_data/MAX/')
subj_max$group = "Human vs MAX"
subj_max$cat = "Human"
# subj_max = read_sim_data_ler("evaluate_model/human_earn_max_sim/RL_nc_dqn_buf_400000_eps_0.2_lr_0.0001", 199)
# subj_max$group = "Human vs MAX"
# subj_max$cat = "Human"

subj_fair = read_mturk_res('human_data/FAIR/')
subj_fair$group = "Human vs FAIR"
subj_fair$cat = "Human"
# subj_fair = read_sim_data_ler("evaluate_model/human_fair_max_sim/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001", 199)
# subj_fair$group = "Human vs FAIR"
# subj_fair$cat = "Human"

gpt_rnd = read_sim_gpt("sim_data/gpt/gpt-3.5-turbo")
gpt_rnd$group = "GPT vs RND"
gpt_rnd$cat = "GPT-3.5"
gpt_adv_max = read_sim_data_real("evaluate_model/earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gpt_adv_max$group = "GPT-3.5 vs MAX"
gpt_adv_max$cat = "GPT-3.5"
gpt_adv_fair = read_sim_data_real("evaluate_model/fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gpt_adv_fair$group = "GPT-3.5 vs FAIR"
gpt_adv_fair$cat = "GPT-3.5"

gpt4_adv_max = read_sim_data_real("evaluate_model/earn_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gpt4_adv_max$group = "GPT-4 vs MAX"
gpt4_adv_max$cat = "GPT-4"
gpt4_adv_fair = read_sim_data_real("evaluate_model/fair_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gpt4_adv_fair$group = "GPT-4 vs FAIR"
gpt4_adv_fair$cat = "GPT-4"

gemini_adv_max = read_sim_data_real("evaluate_model/earn_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gemini_adv_max$group = "Gemini-1.5 vs MAX"
gemini_adv_max$cat = "Gemini-1.5"
gemini_adv_fair = read_sim_data_real("evaluate_model/fair_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gemini_adv_fair$group = "Gemini-1.5 vs FAIR"
gemini_adv_fair$cat = "Gemini-1.5"

deepseek_adv_max = read_sim_data_real("evaluate_model/earn_max_deepseek/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
deepseek_adv_max$group = "DeepSeek-V3 vs MAX"
deepseek_adv_max$cat = "DeepSeek-V3"
deepseek_adv_fair = read_sim_data_real("evaluate_model/fair_max_deepseek/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05", 49)
deepseek_adv_fair$group = "DeepSeek-V3 vs FAIR"
deepseek_adv_fair$cat = "DeepSeek-V3"

# gpt_adv_max = read_sim_data_ler("evaluate_model/earn_max_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
# gpt_adv_max$group = "GPT-3.5 vs MAX"
# gpt_adv_max$cat = "GPT-3.5"
# gpt_adv_fair = read_sim_data_ler("evaluate_model/fair_max_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
# gpt_adv_fair$group = "GPT-3.5 vs FAIR"
# gpt_adv_fair$cat = "GPT-3.5"
# 
# gpt4_adv_max = read_sim_data_ler("evaluate_model/earn_max4_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
# gpt4_adv_max$group = "GPT-4 vs MAX"
# gpt4_adv_max$cat = "GPT-4"
# gpt4_adv_fair = read_sim_data_ler("evaluate_model/fair_max4_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
# gpt4_adv_fair$group = "GPT-4 vs FAIR"
# gpt4_adv_fair$cat = "GPT-4"
# 
# 
# gemini_adv_max = read_sim_data_ler("evaluate_model/earn_max_gemini_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
# gemini_adv_max$group = "Gemini vs MAX"
# gemini_adv_max$cat = "Gemini"
# gemini_adv_fair = read_sim_data_ler("evaluate_model/fair_max_gemini_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
# gemini_adv_fair$group = "Gemini vs FAIR"
# gemini_adv_fair$cat = "Gemini"

# MAX case
alld_rnd = rbind(subj_max[,c("id", "repay_percent", "group", "investment", "cat", "trial")],
                 gpt_adv_max[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)
alld_rnd = rbind(alld_rnd,
                 gpt4_adv_max[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)
alld_rnd = rbind(alld_rnd,
                 gpt4_adv_max[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)
alld_rnd = rbind(alld_rnd,
                 gemini_adv_max[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)
alld_rnd = rbind(alld_rnd,
                 deepseek_adv_max[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)

plot_repay_trial(alld_rnd)
ggsave("plots/repay_trail.pdf", width=19, height=10, unit="cm", useDingbats=FALSE)
plot_investment_trial(alld_rnd)
ggsave("plots/investment_trial.pdf", width=22, height=10, unit="cm", useDingbats=FALSE)


# FAIR case
alld_rnd = rbind(subj_fair[,c("id", "repay_percent", "group", "investment", "cat", "trial")],
                 gpt_adv_fair[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)
alld_rnd = rbind(alld_rnd,
                 gpt4_adv_fair[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)
alld_rnd = rbind(alld_rnd,
                 gemini_adv_fair[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)
alld_rnd = rbind(alld_rnd,
                 deepseek_adv_fair[,c("id", "repay_percent", "group", "investment", "cat", "trial")]
)
plot_repay_trial(alld_rnd)
ggsave("plots/fair_repay_trail.pdf", width=19, height=10, unit="cm", useDingbats=FALSE)

plot_investment_trial(alld_rnd)
ggsave("plots/fair_investment_trial.pdf", width=22, height=10, unit="cm", useDingbats=FALSE)


