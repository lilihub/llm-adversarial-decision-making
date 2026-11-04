
library(ggplot2)
dd = list()
indx = 1

folder_path <- "human_data/1_6_vs_2_6"
# folder_path <- "sim_data/gpt/gpt-3.5-turbo"
# folder_path <- "sim_data/gpt/gpt-4-turbo"
# folder_path <- "sim_data/gemini/gemini_1.5"
# folder_path <- "sim_data/deepseek/deepseek-chat"
csv_files <- list.files(path = folder_path, pattern = ".csv", full.names = TRUE)
files <- csv_files[3:4]
for (file in files) {
  data <- read.csv(file)
   
  # for human data
  data$is_biased_choice <- trimws(data$is_biased_choice)
  data <- data %>% mutate(action = ifelse(is_biased_choice == "true", 1, 0),
                          reward = observed_reward,
                          X = trial_number)
  # for LLM data X was encoded as 0 and Y was encoded as 1 in the simulation dataset
  # data <- data %>% mutate(action = ifelse(action == 0, 1, 0))
  
  data$ev = file
  dd[[indx]] =data
  indx = indx + 1
}

require(data.table)
data = rbindlist(dd)
action_levels = levels(data$action)
data$ev2 = factor(data$ev, levels=rev(levels(as.factor(data$ev))))


ggplot() +
  scale_color_manual(name="action", values=c("red", "blue")) +
  geom_segment(data = subset(data, reward == 1 & action == 1), aes(x=X, xend=X, y=0.5, yend=0.85), show.legend=FALSE, color="blue") +
  geom_segment(data = subset(data, reward == 1 & action == 0), aes(x=X, xend=X, y=0.15, yend=0.5), show.legend=FALSE, color="red") +
  scale_y_continuous(breaks=c(0, 1), limits = c(0.15,0.85), expand = c(0.05, 0.05)) +
  scale_x_continuous(expand = c(0.02, 0.02)) +
  geom_point(data = data, aes(x=X, color=as.factor(action), y = 0.5), show.legend=FALSE, size=1) +
  theme_bw() +
  theme(
    axis.title.y = element_blank(),
    axis.title.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(), 
    text = element_text(size=16),
    axis.text = element_text(size=16),
    # panel.border = element_blank(),  # Remove panel borders
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    strip.text = element_blank(),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10, unit = "pt")  # Adjust these values as needed
  ) +
  facet_grid(ev ~ .) +
  xlab("trial")
ggsave("plots/human_before_adv.pdf", width=20, height=3.3, unit="cm", useDingbats=FALSE)
# ggsave("plots/gpt-3.5_before_adv.pdf", width=20, height=3.3, unit="cm", useDingbats=FALSE)
# ggsave("plots/gpt-4_before_adv.pdf", width=20, height=3.3, unit="cm", useDingbats=FALSE)
# ggsave("plots/gemini_before_adv.pdf", width=20, height=3.3, unit="cm", useDingbats=FALSE)
# ggsave("plots/deepseek_before_adv.pdf", width=20, height=3.3, unit="cm", useDingbats=FALSE)


