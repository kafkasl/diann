
echo "Evaluating Classifier Based Tagger"
#python ../src/evaluation/matching.py ../data/English_Training/Annotated/ ../results/system/ClassifierBasedTagger

echo "Evaluating CRF Tagger - English"
python ../src/evaluation/matching.py ../data/English_Training/Annotated/ ../results/system/CRFTagger/english

echo "Evaluating CRF Tagger - Spanish"
python ../src/evaluation/matching.py ../data/Spanish_Training/Annotated/ ../results/system/CRFTagger/spanish
