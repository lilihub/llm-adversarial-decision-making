library(ggplot2)
dd = list()
indx = 1
# for (i in 12:13){ #for GPT3.5
# for (i in 1:2){ # for GPT-4
# for (i in 11:12){ # for gemini-1.5
for (i in 4:5){ # for deepseek
  # data = read.csv(paste0("evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gpt-3.5-turbo/events_", i,".csv"))
  # pol = read.csv(paste0("evaluate_model/RNN_sim/gpt-3.5-turbo/policies/policies_", i,".csv"))
  # data = read.csv(paste0("evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gpt-4-turbo/events_", i,".csv"))
  # pol = read.csv(paste0("evaluate_model/RNN_sim/gpt-4-turbo/policies/policies_", i,".csv"))
  # data = read.csv(paste0("evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/gemini_1.5/events_", i,".csv"))
  # pol = read.csv(paste0("evaluate_model/RNN_sim/gemini_1.5/policies/policies_", i,".csv"))
  data = read.csv(paste0("evaluate_model/RNN_adv_sim_400000_eps_0.01_lr_0.001/deepseek-chat/events_", i,".csv"))
  pol = read.csv(paste0("evaluate_model/RNN_sim/deepseek-chat/policies/policies_", i,".csv"))
  
  data$ev = i
  data$pol0 = NA
  data$pol1 = NA
  data$pol0 = pol$X0
  data$pol1 = pol$X1
  dd[[indx]] =data
  indx = indx + 1
}

require(data.table)
data = rbindlist(dd)
action_levels = levels(data$real.model.action)
data$rnn.action = as.character(data$real.model.action)

data$ev2 = factor(data$ev, levels=rev(levels(as.factor(data$ev))))
require(ggplot2)
ggplot() +
  scale_color_manual(name="action", values=c("red", "blue")) +
  geom_ribbon(data = subset(data, T), aes(x=X, ymin=0.5, ymax=pol0), fill="green", alpha=0.5) +
  geom_segment(data = subset(data, r1 == 1), aes(x=X, xend=X, y=0.5, yend=1), show.legend=FALSE, color="blue") +
  geom_segment(data = subset(data, r2 == 1), aes(x=X, xend=X, y=0.5, yend=0), show.legend=FALSE, color="red") +
  scale_y_continuous(breaks=c(0, 0.5, 1), limits = c(0,1), expand = c(0.1, 0.1)) +
  scale_x_continuous(expand = c(0.02, 0.02)) +
  geom_point(data = data, aes(x=X, color=as.factor(rnn.action), y = 0.5), show.legend=FALSE, size=1) +
  theme_bw() +
  theme(
    axis.title.y = element_blank(),
    axis.title.x = element_blank(),
    text = element_text(size=16),
    axis.text = element_text(size=16),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    strip.text = element_blank(),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10, unit = "pt")  # Adjust these values as needed
  ) +
  facet_grid(ev ~ .) +
  xlab("trial")
ggsave("plots/ADV_strategy_DeepSeek.pdf", width=20, height=5, unit="cm", useDingbats=FALSE)


