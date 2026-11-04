library(plyr)
library(ggplot2)
library(wesanderson)
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
plot_trustee_earn = function(data, x_order){
  data_r = ddply(subset(data, T), c("id", "group", "cat"), function(x){data.frame(trustee_earn=sum(x$trustee_earn))})
  dark2_colors <- c("#1b9e77", "#7570b3", "#e7298a", "#d95f02", "#e6ab02")
  ggplot(subset(data_r, T), aes(y = trustee_earn, x =group, fill=cat)) +
    #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("trustee earning") +
    xlab("") +
    theme_bw() +
    scale_fill_manual(values=dark2_colors,
                      limits=c("Human","GPT-3.5", "GPT-4", "Gemini-1.5", "DeepSeek-V3"),
                      labels = c("Human", "GPT-3.5", "GPT-4", "Gemini")) +
    # scale_fill_manual(values = wes_palette("GrandBudapest1", n = 3))+
    scale_x_discrete(limits = x_order) +
    coord_cartesian(ylim=c(0, 380))+
    # blk_theme_grid_hor(legend_position ="none", margins = c(1,1,1,1), rotate_x = F) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
    theme(legend.position="none")+
    theme(axis.title.y = element_text(size = 18), # Increase Y-axis label size and make it bold
          axis.text.x = element_text(size = 14), # Increase X-axis tick text size
          axis.text.y = element_text(size = 14))+ # Increase legend title size
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
}

plot_investor_earn = function(data, x_order){
  data_r = ddply(subset(data, T), c("id", "group", "cat"), function(x){data.frame(investor_earn=sum(x$investor_earn))})
  dark2_colors <- c("#1b9e77", "#7570b3", "#e7298a", "#d95f02", "#e6ab02")
  ggplot(subset(data_r, T), aes(y = investor_earn, x =group, fill=cat)) +
    #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("investor earning") +
    xlab("") +
    theme_bw() +
    scale_fill_manual(values=dark2_colors,
                      limits=c("Human","GPT-3.5", "GPT-4", "Gemini-1.5", "DeepSeek-V3"),
                      labels = c("Human", "GPT-3.5", "GPT-4", "Gemini")) +
    scale_x_discrete(limits = x_order) +
    coord_cartesian(ylim=c(0, 380))+
    
    # blk_theme_grid_hor(legend_position ="none", margins = c(1,1,1,1), rotate_x = F) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
    theme(legend.position="none") +
    theme(axis.title.y = element_text(size = 18), # Increase Y-axis label size and make it bold
          axis.text.x = element_text(size = 14), # Increase X-axis tick text size
          axis.text.y = element_text(size = 14)) + # Increase legend title size
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
}

plot_fairness = function(data, x_order){
  data_r = ddply(subset(data, T), c("id", "group", "cat"), function(x){data.frame(diffs=sum(x$trustee_earn - x$investor_earn))})
  dark2_colors <- c("#1b9e77", "#7570b3", "#e7298a", "#d95f02", "#e6ab02")
  ggplot(subset(data_r, T), aes(y = diffs, x =group, fill=cat)) +
    #  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), fill=col) +
    geom_bar(stat = "summary", fun.y = "mean") +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black",
                 position=position_dodge(.9),  fun.args = list(mult = 1), size=0.2) +
    ylab("trustee earning-investor earning") +
    xlab("") +
    theme_bw() +
    scale_fill_manual(values=dark2_colors,
                      limits=c("Human","GPT-3.5", "GPT-4", "Gemini-1.5", "DeepSeek-V3"),
                      labels = c("Human", "GPT-3.5", "GPT-4", "Gemini")) +
    scale_x_discrete(limits = x_order) +
    # blk_theme_grid_hor(legend_position ="none", margins = c(1,1,1,1), rotate_x = F) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
    theme(legend.position="none")+
    theme(axis.title.y = element_text(size = 18), # Increase Y-axis label size and make it bold
          axis.text.x = element_text(size = 14), # Increase X-axis tick text size
          axis.text.y = element_text(size = 14)) + # Increase legend title size
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
}

subj_rnd = read_mturk_res('human_data/RND/')
subj_rnd$group = "Human vs RND"
subj_rnd$cat = "Human"
subj_fair = read_mturk_res('human_data/FAIR/')
subj_fair$group = "Human vs FAIR"
subj_fair$cat = "Human"
subj_max = read_mturk_res('human_data/MAX/')
subj_max$group = "Human vs MAX"
subj_max$cat = "Human"

gpt_rnd = read_sim_gpt("sim_data/gpt/gpt-3.5-turbo")
gpt_rnd$group = "GPT-3.5 vs RND"
gpt_rnd$cat = "GPT-3.5"
gpt_adv_fair = read_sim_data_real("evaluate_model/fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gpt_adv_fair$group = "GPT-3.5 vs FAIR"
gpt_adv_fair$cat = "GPT-3.5"
# gpt_adv_sim_fair = read_sim_data_ler("evaluate_model/fair_max_sim/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001", 199)
# gpt_adv_sim_fair$group = "GPT-3.5 vs FAIR"
# gpt_adv_sim_fair$cat = "GPT-3.5"
gpt_adv_max = read_sim_data_real("evaluate_model/earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gpt_adv_max$group = "GPT-3.5 vs MAX"
gpt_adv_max$cat = "GPT-3.5"
# gpt_adv_sim_max = read_sim_data_ler("evaluate_model/earn_max_sim/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001", 199)
# gpt_adv_sim_max$group = "GPT-3.5 vs MAX"
# gpt_adv_sim_max$cat = "GPT-3.5"

gpt4_rnd = read_sim_gpt("sim_data/gpt/gpt-4-turbo")
gpt4_rnd$group = "GPT-4 vs RND"
gpt4_rnd$cat = "GPT-4"
gpt4_adv_fair = read_sim_data_real("evaluate_model/fair_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gpt4_adv_fair$group = "GPT-4 vs FAIR"
gpt4_adv_fair$cat = "GPT-4"
# gpt4_adv_sim_fair = read_sim_data_ler("evaluate_model/fair_max4_sim/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05", 199)
# gpt4_adv_sim_fair$group = "GPT-4 vs FAIR"
# gpt4_adv_sim_fair$cat = "GPT-4"
gpt4_adv_max = read_sim_data_real("evaluate_model/earn_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gpt4_adv_max$group = "GPT-4 vs MAX"
gpt4_adv_max$cat = "GPT-4"
# gpt4_adv_sim_max = read_sim_data_ler("evaluate_model/earn_max4_sim/RL_nc_dqn_buf_400000_eps_0.2_lr_0.001", 199)
# gpt4_adv_sim_max$group = "GPT-4 vs MAX"
# gpt4_adv_sim_max$cat = "GPT-4"


gemini_rnd = read_sim_gpt("sim_data/gemini/gemini_1.5")
gemini_rnd$group = "Gemini-1.5 vs RND"
gemini_rnd$cat = "Gemini-1.5"
gemini_adv_max = read_sim_data_real("evaluate_model/earn_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gemini_adv_max$group = "Gemini-1.5 vs MAX"
gemini_adv_max$cat = "Gemini-1.5"
# gemini_adv_sim_max = read_sim_data_ler("evaluate_model/earn_max_gemini_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
# gemini_adv_sim_max$group = "Gemini vs MAX"
# gemini_adv_sim_max$cat = "Gemini"
gemini_adv_fair = read_sim_data_real("evaluate_model/fair_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
gemini_adv_fair$group = "Gemini-1.5 vs FAIR"
gemini_adv_fair$cat = "Gemini-1.5"

deepseek_rnd = read_sim_gpt("sim_data/deepseek/deepseek-chat")
deepseek_rnd$group = "DeepSeek-V3 vs RND"
deepseek_rnd$cat = "DeepSeek-V3"
deepseek_adv_max = read_sim_data_real("evaluate_model/earn_max_deepseek/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
deepseek_adv_max$group = "DeepSeek-V3 vs MAX"
deepseek_adv_max$cat = "DeepSeek-V3"
# gemini_adv_sim_max = read_sim_data_ler("evaluate_model/earn_max_gemini_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
# gemini_adv_sim_max$group = "Gemini vs MAX"
# gemini_adv_sim_max$cat = "Gemini"
deepseek_adv_fair = read_sim_data_real("evaluate_model/fair_max_deepseek/RL_nc_dqn_buf_400000_eps_0.2_lr_1e-05", 49)
deepseek_adv_fair$group = "Deepseek-V3 vs FAIR"
deepseek_adv_fair$cat = "DeepSeek-V3"


alld_rnd = rbind(subj_rnd[,c("id", "trustee_earn", "group", "cat")],
                 subj_max[,c("id", "trustee_earn", "group", "cat")],
                 subj_fair[,c("id", "trustee_earn", "group", "cat")],
                 gpt_rnd[,c("id", "trustee_earn", "group", "cat")],
                 gpt_adv_max[,c("id", "trustee_earn", "group", "cat")],
                 gpt_adv_fair[,c("id", "trustee_earn", "group", "cat")],
                 gpt4_rnd[,c("id", "trustee_earn", "group", "cat")],
                 gpt4_adv_max[,c("id", "trustee_earn", "group", "cat")],
                 gpt4_adv_fair[,c("id", "trustee_earn", "group", "cat")],
                 gemini_rnd[,c("id", "trustee_earn", "group", "cat")],
                 gemini_adv_max[,c("id", "trustee_earn", "group", "cat")],
                 gemini_adv_fair[,c("id", "trustee_earn", "group", "cat")],
                 deepseek_rnd[,c("id", "trustee_earn", "group", "cat")],
                 deepseek_adv_max[,c("id", "trustee_earn", "group", "cat")],
                 deepseek_adv_fair[,c("id", "trustee_earn", "group", "cat")])
plot_trustee_earn(alld_rnd, c(subj_rnd$group[1],
                              subj_fair$group[1],
                              subj_max$group[1],
                              gpt_rnd$group[1],
                              gpt_adv_fair$group[1],
                              gpt_adv_max$group[1],
                              gpt4_rnd$group[1],
                              gpt4_adv_fair$group[1],
                              gpt4_adv_max$group[1],
                              gemini_rnd$group[1],
                              gemini_adv_fair$group[1],
                              gemini_adv_max$group[1],
                              deepseek_rnd$group[1],
                              deepseek_adv_fair$group[1],
                              deepseek_adv_max$group[1]))
ggsave("plots/trustee_earn.pdf", width=14, height=16, unit="cm", useDingbats=FALSE)


# Assuming data_r is created as in your function
data_r <- ddply(subset(alld_rnd, T), c("id", "group", "cat"), 
                function(x) data.frame(trustee_earn = sum(x$trustee_earn)))
# Calculate mean trustee earning for each group and cat
group_means <- ddply(data_r, c("group", "cat"), summarise, mean_earn = mean(trustee_earn))
print(group_means)

alld_rnd = rbind(subj_rnd[,c("id", "investor_earn", "group", "cat")],
                 subj_max[,c("id", "investor_earn", "group", "cat")],
                 subj_fair[,c("id", "investor_earn", "group", "cat")],
                 gpt_rnd[,c("id", "investor_earn", "group", "cat")],
                 gpt_adv_fair[,c("id", "investor_earn", "group", "cat")],
                 gpt_adv_max[,c("id", "investor_earn", "group", "cat")],
                 gpt4_rnd[,c("id", "investor_earn", "group", "cat")],
                 gpt4_adv_fair[,c("id", "investor_earn", "group", "cat")],
                 gpt4_adv_max[,c("id", "investor_earn", "group", "cat")],
                 gemini_rnd[,c("id", "investor_earn", "group", "cat")],
                 gemini_adv_max[,c("id", "investor_earn", "group", "cat")],
                 gemini_adv_fair[,c("id", "investor_earn", "group", "cat")],
                 deepseek_rnd[,c("id", "investor_earn", "group", "cat")],
                 deepseek_adv_max[,c("id", "investor_earn", "group", "cat")],
                 deepseek_adv_fair[,c("id", "investor_earn", "group", "cat")])
plot_investor_earn(alld_rnd, c(subj_rnd$group[1],
                               subj_fair$group[1],
                               subj_max$group[1],
                               gpt_rnd$group[1],
                               gpt_adv_fair$group[1],
                               gpt_adv_max$group[1],
                               gpt4_rnd$group[1],
                               gpt4_adv_fair$group[1],
                               gpt4_adv_max$group[1],
                               gemini_rnd$group[1],
                               gemini_adv_fair$group[1],
                               gemini_adv_max$group[1],
                               deepseek_rnd$group[1],
                               deepseek_adv_fair$group[1],
                               deepseek_adv_max$group[1]))

ggsave("plots/investor_earn.pdf", width=14, height=16, unit="cm", useDingbats=FALSE)

alld_rnd = rbind(subj_rnd[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 subj_max[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 subj_fair[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gpt_rnd[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gpt_adv_max[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gpt_adv_fair[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gpt4_rnd[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gpt4_adv_max[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gpt4_adv_fair[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gemini_rnd[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gemini_adv_max[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 gemini_adv_fair[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 deepseek_rnd[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 deepseek_adv_max[,c("id", "trustee_earn", "group", "investor_earn", "cat")],
                 deepseek_adv_fair[,c("id", "trustee_earn", "group", "investor_earn", "cat")])
plot_fairness(alld_rnd, c(subj_rnd$group[1],
                          subj_fair$group[1],
                          subj_max$group[1],
                          gpt_rnd$group[1],
                          gpt_adv_fair$group[1],
                          gpt_adv_max$group[1],
                          gpt4_rnd$group[1],
                          gpt4_adv_fair$group[1],
                          gpt4_adv_max$group[1],
                          gemini_rnd$group[1],
                          gemini_adv_fair$group[1],
                          gemini_adv_max$group[1],
                          deepseek_rnd$group[1],
                          deepseek_adv_fair$group[1],
                          deepseek_adv_max$group[1]))
ggsave("plots/earning_gap.pdf", width=14, height=16, unit="cm", useDingbats=FALSE)

