plot_adv_strategy = function(data){
  data_tile = ddply(subset(data, investment != 0), c("trial", "investement.discr"), function(x){data.frame(adv_mean=mean(x$repay_percent))})
  ggplot() +
    geom_tile(data=data_tile,
              aes(x=as.factor(trial), y = as.factor(investement.discr * 20), fill= adv_mean),
    ) +
    xlab("") +
    ylab("") +
    scale_fill_continuous(type = "viridis", name="advesary action (% repayment)",
                          limits=c(0, 100), breaks = c(0, 25, 50, 75,100)) +
    theme_bw() +
    theme(legend.position="bottom", legend.box = "horizontal") +
    theme(text = element_text(size=8))
}

subj_rnd = read_mturk_res('human_data/RND/')
subj_rnd$group = "Human vs RND"
subj_rnd$cat = "RND"
subj_max = read_mturk_res('human_data/MAX/')
subj_max$group = "Human vs MAX"
subj_max$cat = "MAX"
subj_fair = read_mturk_res('human_data/FAIR/')
subj_fair$group = "Human vs FAIR"
subj_fair$cat = "FAIR"

# subj_max = read_sim_data_ler("evaluate_model/human_earn_max_sim/RL_nc_dqn_buf_400000_eps_0.2_lr_0.0001", 199)
# subj_max$group = "Human vs MAX"
# subj_max$cat = "Human"
# subj_fair = read_sim_data_ler("evaluate_model/human_fair_max_sim/RL_nc_dqn_buf_400000_eps_0.1_lr_0.0001", 199)
# subj_fair$group = "Human vs FAIR"
# subj_fair$cat = "Human"

# gpt_adv_max = read_sim_data_real("evaluate_model/earn_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
# gpt_adv_max$group = "GPT vs MAX"
# gpt_adv_max$cat = "MAX"
# gpt_adv_fair = read_sim_data_real("evaluate_model/fair_max/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
# gpt_adv_fair$group = "GPT vs FAIR"
# gpt_adv_fair$cat = "FAIR"

gpt_adv_max = read_sim_data_ler("evaluate_model/earn_max_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
gpt_adv_max$group = "GPT vs MAX"
gpt_adv_max$cat = "MAX"
gpt_adv_fair = read_sim_data_ler("evaluate_model/fair_max_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
gpt_adv_fair$group = "GPT vs FAIR"
gpt_adv_fair$cat = "FAIR"


# gpt4_adv_max = read_sim_data_real("evaluate_model/earn_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
# gpt4_adv_max$group = "GPT-4 vs MAX"
# gpt4_adv_max$cat = "GPT-4"
# gpt4_adv_fair = read_sim_data_real("evaluate_model/fair_max4/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
# gpt4_adv_fair$group = "GPT-4 vs FAIR"
# gpt4_adv_fair$cat = "GPT-4"

gpt4_adv_max = read_sim_data_ler("evaluate_model/earn_max4_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
gpt4_adv_max$group = "GPT-4 vs MAX"
gpt4_adv_max$cat = "GPT-4"
gpt4_adv_fair = read_sim_data_ler("evaluate_model/fair_max4_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
gpt4_adv_fair$group = "GPT-4 vs FAIR"
gpt4_adv_fair$cat = "GPT-4"


# gemini_adv_max = read_sim_data_real("evaluate_model/earn_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
# gemini_adv_max$group = "Gemini vs MAX"
# gemini_adv_max$cat = "Gemini"
# gemini_adv_fair = read_sim_data_real("evaluate_model/fair_max_gemini/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 49)
# gemini_adv_fair$group = "Gemini vs FAIR"
# gemini_adv_fair$cat = "Gemini"

gemini_adv_max = read_sim_data_ler("evaluate_model/earn_max_gemini_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
gemini_adv_max$group = "Gemini vs MAX"
gemini_adv_max$cat = "Gemini"
gemini_adv_fair = read_sim_data_ler("evaluate_model/fair_max_gemini_sim/RL_nc_dqn_buf_400000_eps_0.01_lr_0.001", 199)
gemini_adv_fair$group = "Gemini vs FAIR"
gemini_adv_fair$cat = "Gemini"
subj_rnd = read_mturk_res('data/RND/')
subj_rnd$group = "Human vs RND"
subj_rnd$cat = "RND"
subj_max = read_mturk_res('data/MAX/')
subj_max$group = "Human vs MAX"
subj_max$cat = "MAX"
subj_fair = read_mturk_res('data/FAIR/')
subj_fair$group = "Human vs FAIR"
subj_fair$cat = "FAIR"
