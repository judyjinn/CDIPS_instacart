# CDIPS_instacart

Berkeley Data Science Workshop
July 15 - Aug 5, 2017

3 week intensive workshop for learning methods in Data Science. Team based projects with a final presentation of results at conclusion of workshop.

Team used open source data provided by Instacart. https://www.instacart.com/datasets/grocery-shopping-2017 200,000+ users with over 3 million orders and 49,000+ unique products in the data set. Each team member in the group chose an algorthm for analyzing the data.

Goal: Predict products which users will reorder in their next order from their previous purchasing history.

Chosen method: Random forest trained using manually extracted features.

forest_instacart.py is the script for training a random forest. It also contains the script for cross-fold validation to determine optimal forest size and a F1 score calculator

generate_sets.py contains script for generating features from Instacart data

graph_insta.py was used to produce a few simple graphs

2017 CDIPS Presentation.pdf are slides from the presentation at the conclusion of the workshop.
