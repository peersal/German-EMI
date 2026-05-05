library(tidyverse)

prodeminfo_data_raw <- read_delim("validation/keyword/PRODEMINFO_-_German_clean.csv",
                                  delim = ";", escape_double = FALSE, trim_ws = TRUE)

# demographics
age <- as.numeric(prodeminfo_data_raw[[3]][-1])
c(mean = mean(age, na.rm = TRUE),
  sd   = sd(age, na.rm = TRUE))

gender <- as.numeric(prodeminfo_data_raw[[4]][-1])
table(gender)


intuition_data <- prodeminfo_data_raw %>%
  select(-c(Q1, Age, Gender)) %>%
  select(matches("#1|ID")) %>%
  setNames(unlist(.[1, ])) %>%
  slice(-1) %>%
  rename(id = 1) %>%
  mutate(across(2:114, as.numeric)) %>%
  pivot_longer(!id) %>%
  mutate(rate_type = "intuition")

evidence_data <- prodeminfo_data_raw %>%
  select(-c(Q1, Age, Gender)) %>%
  select(matches("#2|ID")) %>%
  setNames(unlist(.[1, ])) %>%
  slice(-1) %>%
  rename(id = 1) %>%
  mutate(across(2:114, as.numeric)) %>%
  pivot_longer(!id) %>%
  mutate(rate_type = "evidence")

binded_long_df <- rbind(intuition_data, evidence_data) %>%
  rename(keyword = name)


# =====================
# VISUALIZATION + SAVE
# =====================

# A–L keywords
p1 <- ggplot(binded_long_df %>% filter(grepl("^[a-lA-L]", keyword))) +
  aes(x = rate_type, y = value, fill = rate_type) +
  geom_boxplot() +
  scale_fill_hue(direction = 1) +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white", color = NA)) +
  facet_wrap(vars(keyword))

ggsave("results/validation/keywords/plot_a_l.png", plot = p1, width = 12, height = 10)


# M–Z keywords
p2 <- ggplot(binded_long_df %>% filter(grepl("^[m-zM-Z]", keyword))) +
  aes(x = rate_type, y = value, fill = rate_type) +
  geom_boxplot() +
  scale_fill_hue(direction = 1) +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white", color = NA)) +
  facet_wrap(vars(keyword))

ggsave("results/validation/keywords/plot_m_z.png", plot = p2, width = 12, height = 10)


# =====================
# T-TESTS
# =====================

t_test_results <- binded_long_df %>%
  group_by(keyword) %>%
  rstatix::t_test(value ~ rate_type, paired = TRUE,
                  p.adjust.method = "holm", detailed = TRUE) %>%
  mutate(sign = ifelse(p < 0.05, "yes", "no"),
         component = ifelse(statistic > 0, "evidence", "intuition"))

intuition_keywords <- t_test_results %>%
  filter(sign == "yes" & component == "intuition") %>%
  select(keyword) %>%
  arrange() %>%
  rename(intuition = 1)

evidence_keywords <- t_test_results %>%
  filter(sign == "yes" & component == "evidence") %>%
  select(keyword) %>%
  arrange() %>%
  rename(evidence = 1)

#write_excel_csv(t_test_results, "PRODEMINFO_Italian_t_test.csv")
#write_excel_csv(intuition_keywords, "PRODEMINFO_Italian_int.csv")
#write_excel_csv(evidence_keywords, "PRODEMINFO_Italian_evi.csv")
