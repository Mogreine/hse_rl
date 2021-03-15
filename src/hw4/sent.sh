rm agent.pt
cp ~/Downloads/best_agent.pt .
mv best_agent.pt agent.pt
rm submission.zip
zip submission.zip agent.pt agent.py train.py