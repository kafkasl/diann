
echo "Evaluating Classifier Based Tagger"
python ../src/evaluation/matching.py ../data/English_Training/Annotated/ ../results/system/ClassifierBasedTagger

echo "Evaluating CRF Tagger"
python ../src/evaluation/matching.py ../data/English_Training/Annotated/ ../results/system/CRFTagger
