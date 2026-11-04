library(ggplot2)
library(data.table)
library(tidyr)
library(tidyverse)
source('R/utils/helper.R')
source("R/utils/summary_utils.R")

############################ GPT-3.5 data ##########################
gpt_data = read.csv('sim_data/gpt/gpt-3.5-turbo.csv')
gpt_data$X = NULL
colnames(gpt_data) = c("subjID", "trial", "choice", "outcome", "probsX", "probsY")

gpt_data_df <- as.data.table(gpt_data)
gpt_counts <- gpt_data_df %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
gpt_counts['condition'] = "GPT-3.5"
gpt_counts['group'] = "Before"
mean(gpt_counts$X)
############################ GPT-3.5_vs_ADV ###################################
sim_gpt_list <- list()
for (i in 1:49){
  data = read.csv(paste0("evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gpt-3.5-turbo/events_", i,".csv"))
  sim_gpt_list <- append(sim_gpt_list, list(data))
}

sim_gpt <- do.call(rbind, sim_gpt_list)
sim_gpt['subjID'] = rep(1:49, each = 100)
sim_gpt$subjID <- paste(sim_gpt$subjID, "gpt-3.5", sep="-")

# colnames(df) = c("subjID", "trial", "choice", "outcome", "probsX", "probsY")
sim_gpt['choice'] =  factor(sim_gpt$real.model.action, 
                            levels = c("[[1. 0.]]", "[[0. 1.]]"),
                            labels = c("0", "1"))
sim_gpt <- sim_gpt %>% 
  mutate(outcome = case_when(choice==1 & r2==0 ~ 0,
                             choice==1 & r2==1 ~ 1,
                             choice==0 & r1==0 ~ 0,
                             choice==0 & r1==1 ~ 1)) 

sim_gpt_df <- as.data.table(sim_gpt)
sim_gpt_counts <- sim_gpt_df %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
sim_gpt_counts['condition'] = "GPT-3.5"
sim_gpt_counts['group'] = "After"
mean(sim_gpt_counts$X)
#############################  human data ###########################
human_dynamic = read.csv('human_data/merged_dynamic.csv') ## 97 subjects
human_static = read.csv("human_data/merged_static.csv") ## 387 subjects
human_combined = rbind(human_dynamic, human_static)
human_combined$X = NULL
colnames(human_combined) = c("choice", "outcome", "subjID", "block", "schedule_type")
human_counts <- human_combined %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
human_counts['condition'] = "Human"
human_counts['group'] = "Before"

############################### human vs. ADV ######################
dirs = list.files("human_data/human_adv_data/", full.names = TRUE, recursive = FALSE)
require(ramify)
sim_human_list = list()
for (d in dirs){
  data = read.csv(d)
  sim_human_list <- append(sim_human_list, list(data))
}
sim_human <- do.call(rbind, sim_human_list)
sim_human['subjID'] = rep(1:157, each = 100)
sim_human_table <- as.data.table(sim_human)

sim_human_counts <- sim_human_table %>%
                       group_by(subjID) %>%
                       summarise(X = sum(is_biased_choice == " true"),
                                 Y = sum(is_biased_choice == " false"))
sim_human_counts['condition'] = "Human"
sim_human_counts['group'] = "After"

############################ GPT-4 data ##########################
gpt4_data = read.csv('sim_data/gpt/gpt-4-turbo.csv')
gpt4_data$X = NULL
colnames(gpt4_data) = c("subjID", "trial", "choice", "outcome", "probsX", "probsY")

gpt4_data_df <- as.data.table(gpt4_data)
gpt4_counts <- gpt4_data_df %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
gpt4_counts['condition'] = "GPT-4"
gpt4_counts['group'] = "Before"
mean(gpt4_counts$X)
############################ GPT-4_vs_ADV ###################################
sim_gpt4_list <- list()
for (i in 0:49){
  data = read.csv(paste0("evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gpt-4-turbo/events_", i,".csv"))
  sim_gpt4_list <- append(sim_gpt4_list, list(data))
}

sim_gpt4 <- do.call(rbind, sim_gpt4_list)
sim_gpt4['subjID'] = rep(0:49, each = 100)
sim_gpt4$subjID <- paste(sim_gpt4$subjID, "gpt-4", sep="-")
# colnames(df) = c("subjID", "trial", "choice", "outcome", "probsX", "probsY")
sim_gpt4['choice'] =  factor(sim_gpt4$real.model.action, 
                            levels = c("[[1. 0.]]", "[[0. 1.]]"),
                            labels = c("0", "1"))
sim_gpt4 <- sim_gpt4 %>% 
  mutate(outcome = case_when(choice==1 & r2==0 ~ 0,
                             choice==1 & r2==1 ~ 1,
                             choice==0 & r1==0 ~ 0,
                             choice==0 & r1==1 ~ 1)) 

sim_gpt4_df <- as.data.table(sim_gpt4)
sim_gpt4_counts <- sim_gpt4_df %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
sim_gpt4_counts['condition'] = "GPT-4"
sim_gpt4_counts['group'] = "After"
mean(sim_gpt4_counts$X)

############################ Gemini data ##########################
gemini_data = read.csv('sim_data/gemini/gemini_1.5.csv')
gemini_data$X = NULL
colnames(gemini_data) = c("subjID", "trial", "choice", "outcome", "probsX", "probsY")

gemini_data_df <- as.data.table(gemini_data)
gemini_counts <- gemini_data_df %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
gemini_counts['condition'] = "Gemini_1.5"
gemini_counts['group'] = "Before"
mean(gemini_counts$X)
############################ Gemini_vs_ADV ###################################
sim_gemini_list <- list()
for (i in 0:49){
  data = read.csv(paste0("evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gemini_1.5/events_", i,".csv"))
  sim_gemini_list <- append(sim_gemini_list, list(data))
}

sim_gemini <- do.call(rbind, sim_gemini_list)
sim_gemini['subjID'] = rep(0:49, each = 100)
sim_gemini$subjID <- paste(sim_gemini$subjID, "gemini_1.5", sep="-")
# colnames(df) = c("subjID", "trial", "choice", "outcome", "probsX", "probsY")
sim_gemini['choice'] =  factor(sim_gemini$real.model.action, 
                             levels = c("[[1. 0.]]", "[[0. 1.]]"),
                             labels = c("0", "1"))
sim_gemini <- sim_gemini %>% 
  mutate(outcome = case_when(choice==1 & r2==0 ~ 0,
                             choice==1 & r2==1 ~ 1,
                             choice==0 & r1==0 ~ 0,
                             choice==0 & r1==1 ~ 1)) 

sim_gemini_df <- as.data.table(sim_gemini)
sim_gemini_counts <- sim_gemini_df %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
sim_gemini_counts['condition'] = "Gemini_1.5"
sim_gemini_counts['group'] = "After"
mean(sim_gemini_counts$X)


############################ Deepseek data ##########################
deepseek_data = read.csv('sim_data/deepseek/deepseek-chat.csv')
deepseek_data$X = NULL
colnames(deepseek_data) = c("subjID", "trial", "choice", "outcome", "probsX", "probsY")

deepseek_data_df <- as.data.table(deepseek_data)
deepseek_counts <- deepseek_data_df %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
deepseek_counts['condition'] = "DeepSeek-V3"
deepseek_counts['group'] = "Before"
mean(deepseek_counts$X)
############################ DeepSeek_vs_ADV ###################################
sim_deepseek_list <- list()
for (i in 1:49){
  data = read.csv(paste0("evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/deepseek-chat/events_", i,".csv"))
  sim_deepseek_list <- append(sim_deepseek_list, list(data))
}

sim_deepseek <- do.call(rbind, sim_deepseek_list)
sim_deepseek['subjID'] = rep(1:49, each = 100)
sim_deepseek$subjID <- paste(sim_deepseek$subjID, "DeepSeek-V3", sep="-")
# colnames(df) = c("subjID", "trial", "choice", "outcome", "probsX", "probsY")
sim_deepseek['choice'] =  factor(sim_deepseek$real.model.action, 
                               levels = c("[[1. 0.]]", "[[0. 1.]]"),
                               labels = c("0", "1"))
sim_deepseek <- sim_deepseek %>% 
  mutate(outcome = case_when(choice==1 & r2==0 ~ 0,
                             choice==1 & r2==1 ~ 1,
                             choice==0 & r1==0 ~ 0,
                             choice==0 & r1==1 ~ 1)) 

sim_deepseek_df <- as.data.table(sim_deepseek)
sim_deepseek_counts <- sim_deepseek_df %>%
  group_by(subjID) %>%
  summarise(X = sum(choice == "0"),
            Y = sum(choice == "1"))
sim_deepseek_counts['condition'] = "DeepSeek-V3"
sim_deepseek_counts['group'] = "After"
mean(sim_deepseek_counts$X)


merge_df = rbind(human_counts, sim_human_counts)
merge_df = rbind(merge_df, gpt_counts)
merge_df = rbind(merge_df, sim_gpt_counts)
merge_df = rbind(merge_df, gpt4_counts)
merge_df = rbind(merge_df, sim_gpt4_counts)
merge_df = rbind(merge_df, gemini_counts)
merge_df = rbind(merge_df, sim_gemini_counts)
merge_df = rbind(merge_df, sim_deepseek_counts)
merge_df = rbind(merge_df, deepseek_counts)


merge_df['combine'] = paste0(merge_df$condition, merge_df$group, sep="")
# write.csv(merge_df, "R/merge_df.csv", row.names = FALSE)
dark2_colors <- c("#e6ab02", "#d95f02", "#7570b3", "#e7298a", "#1b9e77")
ggplot(merge_df, aes(x=combine, y=X, fill=condition))+
  # geom_boxplot()+
  stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2) +
  # stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
  #              position=position_dodge(.9),  fun.args = list(mult = 1)) +
  # geom_point(color="black", position=position_nudge(x = -0.2, y = 0), alpha=0.5, size=1) +
  # geom_line(aes(group=subjID), alpha= 0.1, position=position_nudge(x = -0.2, y = 0), size=0.7) +
  geom_jitter(shape=1, position=position_jitter(0.1))+
  scale_fill_manual(values=dark2_colors) +   # Purple from Dark2
  scale_x_discrete(limits = c("HumanBefore", "HumanAfter",
                              "GPT-3.5Before", "GPT-3.5After", 
                              "GPT-4Before", "GPT-4After",
                              "Gemini_1.5Before", "Gemini_1.5After",
                              "DeepSeek-V3Before", "DeepSeek-V3After"),
                   labels = c("Before", "After", "Before", "After", 
                              "Before", "After", "Before", "After",
                              "Before", "After")) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 3.0))+
  theme_no_legend_x_title()+
  theme(axis.title.y = element_text(size = 18), # Increase Y-axis label size and make it bold
        axis.text.x = element_text(size = 16), # Increase X-axis tick text size
        axis.text.y = element_text(size = 16)) + # Increase legend title size
  ylab("The target option%")
ggsave("plots/BA_ADV.pdf", width=20, height=25, unit="cm", useDingbats=FALSE)



