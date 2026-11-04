## convert dataframe to list
frame2list <- function(df){
  DT_trials <- df[, .N, by = "subjID"]
  subjs     <- DT_trials$subjID
  n_subj    <- length(subjs)
  t_subjs   <- DT_trials$N
  t_max     <- max(t_subjs)
  
  # Initialize (model-specific) data arrays
  choice  <- array(-1, c(n_subj, t_max))   ###recycle -1 array n_subj rows and t_max columns
  outcome <- array(-1, c(n_subj, t_max))
  
  # Write from raw_data to the data arrays
  for (i in 1:n_subj) {
    subj <- subjs[i]
    t <- t_subjs[i]
    DT_subj <- df[df$subjID == subj]
    
    choice[i, 1:t]  <- DT_subj$choice
    outcome[i, 1:t] <- DT_subj$outcome
  }
  # Wrap into a list for Stan
  data_list <- list(
    N       = n_subj,
    T       = t_max,
    Tsubj   = t_subjs,
    choice  = choice,
    outcome = outcome
  )
  return(data_list)
}
summary_plot <- function(df, stat1, stat2, ylabel){
  ggplot(df, aes(x = option, y = stat_name, fill = option)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2) +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
                 position=position_dodge(.9),  fun.args = list(mult = 1)) +
    # xlab(c("Condition")) +  
    ylab(ylabel) +
    geom_point(color="black", position=position_nudge(x = -0.2, y = 0), alpha=0.5, size=1) +
    # geom_line(aes(group=subjID), alpha= 0.1, position=position_nudge(x = -0.2, y = 0), size=0.7) +
    scale_fill_brewer(name = "", palette=palette_mode, breaks = c(stat1, stat2), labels = c("X", "Y"))+
    blk_theme_no_grid_nostrip()
  # facet_grid(. ~ src) +
  # guides(fill = guide_legend(keywidth = 0.5, keyheight = 1.0))
  
}
loss_shift_fun <- function(list_data){
  choice = list_data$choice
  outcome = list_data$outcome
  n_subj = nrow(choice)
  last_choice = cbind(rep(NA, n_subj), choice[, 1:ncol(choice)-1])
  shift = last_choice != choice
  last_loss = cbind(rep(NA, n_subj), outcome[, 1:ncol(outcome)-1])
  loss_shift = (shift==1 & last_loss==0)[,2:ncol(choice)]
  loss_no_shift = (shift==0 & last_loss==0)[,2:ncol(choice)]
  reward_shift = (shift==1 & last_loss==1)[, 2:ncol(choice)]
  reward_no_shift = (shift==0 & last_loss==1)[, 2:ncol(choice)]
  
  trial_loss_shift = colSums(loss_shift)
  trial_loss_no_shift = colSums(loss_no_shift)
  trial_reward_shift = colSums(reward_shift)
  trial_reward_no_shift = colSums(reward_no_shift)
  trial_loss_shift_percen = trial_loss_shift/(trial_loss_shift+trial_loss_no_shift)
  trial_reward_shift_percen = trial_reward_shift/(trial_reward_shift+trial_reward_no_shift)
  
  subj_loss_shift = rowSums(loss_shift)
  subj_loss_no_shift = rowSums(loss_no_shift)
  subj_loss_shift_percen = subj_loss_shift/(subj_loss_shift+subj_loss_no_shift)
  
  subj_reward_shift = rowSums(reward_shift)
  subj_reward_no_shift = rowSums(reward_no_shift)
  subj_reward_shift_percen = subj_reward_shift/(subj_reward_shift+subj_reward_no_shift)
  
  return_list = list(loss_mean_trial = trial_loss_shift_percen, 
                     reward_mean_trial = trial_reward_shift_percen,
                     loss_mean_sub = subj_loss_shift_percen,
                     reward_mean_sub = subj_reward_shift_percen)
}

subject_plot <- function(list, legend1, legend2, ylabel){
  subject_df = data.frame(subjID = as.factor(rep(seq(1, length(list$loss_mean_sub), 1), 2)),
                          sub_mean = c(list$loss_mean_sub, 
                                       list$reward_mean_sub), 
                          condition = c(rep(legend1, each=length(list$loss_mean_sub)),
                                        rep(legend2, each=length(list$reward_mean_sub))))
  
  p = ggplot(subject_df, aes(x=condition, y=sub_mean, fill=condition))+
    # geom_boxplot()+
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(), color="black", size=0.2) +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour=error_bar_colour,
                 position=position_dodge(.9),  fun.args = list(mult = 1)) +
    # geom_point(color="black", position=position_nudge(x = -0.2, y = 0), alpha=0.5, size=1) +
    geom_line(aes(group=subjID), alpha= 0.1, position=position_nudge(x = -0.2, y = 0), size=0.7) +
    geom_jitter(shape=1, position=position_jitter(0.25))+
    scale_fill_brewer(name = "", palette=palette_mode,
                      breaks = c(legend1, legend2),
                      labels = c("no_reward_shift", "reward_shift"))+
    blk_theme_no_grid()+
    # ggtitle(title) +
    ylab(ylabel)
  return(p)
}

