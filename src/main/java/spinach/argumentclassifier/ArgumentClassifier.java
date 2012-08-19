package spinach.argumentclassifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;
import spinach.classify.Classifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.*;

public abstract class ArgumentClassifier {

    protected final Classifier classifier;
    protected final ArgumentFeatureGenerator featureGenerator;

    public ArgumentClassifier(Classifier classifier, ArgumentFeatureGenerator featureGenerator){
        this.classifier = classifier;
        this.featureGenerator = featureGenerator;
    }

    public abstract SemanticFrameSet framesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates);

    public static List<Token> argumentCandidates(TokenSentence sentence, Token predicate) {
        List<Token> argumentCandidates = new ArrayList<Token>();
        Token currentHead = predicate;
        while (currentHead != null){
            argumentCandidates.addAll(sentence.getChildren(currentHead));
            if(currentHead.headSentenceIndex < 0){
                argumentCandidates.add(currentHead);
                break;
            }
            currentHead = sentence.getParent(currentHead);
        }

        Collections.sort(argumentCandidates, new Comparator<Token>() {
            public int compare(Token t1, Token t2){
                return new Integer(t1.sentenceIndex).
                        compareTo(t2.sentenceIndex);
            }
        });

        return argumentCandidates;
    }

    protected Counter<String> argClassScores(SemanticFrameSet frameSet, Token possibleArg, Token predicate) {
        return classifier.scoresOf(featureGenerator.datumFrom(frameSet, possibleArg, predicate));
    }

    protected static List<String> sortArgLabels(Counter<String> argCounter) {
        List<Map.Entry<String, Double>> list = new LinkedList<Map.Entry<String, Double>>(argCounter.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                return (o2.getValue())
                        .compareTo(o1.getValue());
            }
        });
        List<String> sortedLabels = new ArrayList<String>();
        for (Map.Entry<String, Double> entry : list)
            sortedLabels.add(entry.getKey());
        return sortedLabels;
    }

    public Dataset<String, String> goldDataset(SemanticFrameSet goldFrames){
        Dataset<String, String> dataset = new Dataset<String, String>();
        for (Token predicate : goldFrames.getPredicateList()){
            for (Token argument : argumentCandidates(goldFrames, predicate)){
                BasicDatum<String, String> datum = (BasicDatum<String, String>) featureGenerator.datumFrom(goldFrames, argument, predicate);
                String label = "NIL";
                for (Pair<Token, String> p : goldFrames.argumentsOf(predicate)){
                    if (p.first().equals(argument)){
                        label = p.second();
                        break;
                    }
                }
                datum.setLabel(label);
                dataset.add(datum);
            }
        }

        return dataset;
    }

    public Dataset<String, String> goldDataset(Collection<SemanticFrameSet> goldFrameSets){
        Dataset<String, String> dataset = new Dataset<String, String>();
        for (SemanticFrameSet frameSet : goldFrameSets)
            dataset.addAll(goldDataset(frameSet));

        dataset.applyFeatureCountThreshold(3);

        return dataset;
    }


}
