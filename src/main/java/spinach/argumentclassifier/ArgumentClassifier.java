package spinach.argumentclassifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.stats.Counter;
import spinach.argumentclassifier.featuregen.ArgumentFeatureGenerator;
import spinach.classifier.PerceptronClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.*;

public abstract class ArgumentClassifier {

    protected final PerceptronClassifier classifier;
    protected final ArgumentFeatureGenerator featureGenerator;

    public final static String NIL_LABEL = "NIL";

    public ArgumentClassifier(PerceptronClassifier classifier, ArgumentFeatureGenerator featureGenerator) {
        this.classifier = classifier;
        this.featureGenerator = featureGenerator;
    }

    public abstract SemanticFrameSet framesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates);

    public abstract SemanticFrameSet trainingFramesWithArguments(TokenSentenceAndPredicates sentenceAndPredicates);

    public static List<Token> argumentCandidates(TokenSentence sentence, Token predicate) {
        List<Token> argumentCandidates = new ArrayList<Token>();
        Token currentHead = predicate;
        while (currentHead != null) {
            argumentCandidates.addAll(sentence.getChildren(currentHead));
            if (currentHead.headSentenceIndex < 0) {
                argumentCandidates.add(currentHead);
                break;
            }
            currentHead = sentence.getParent(currentHead);
        }

        Collections.sort(argumentCandidates, new Comparator<Token>() {
            public int compare(Token t1, Token t2) {
                return new Integer(t1.sentenceIndex).
                        compareTo(t2.sentenceIndex);
            }
        });

        return argumentCandidates;
    }

    protected Counter<String> argClassScores(SemanticFrameSet frameSet, Token possibleArg, Token predicate) {
        return classifier.scoresOf(featureGenerator.datumFrom(frameSet, possibleArg, predicate));
    }

    protected Counter<String> trainingArgClassScores(SemanticFrameSet frameSet, Token possibleArg, Token predicate) {
        return classifier.trainingScores(featureGenerator.datumFrom(frameSet, possibleArg, predicate));
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

    public Dataset<String, String> datasetFrom(SemanticFrameSet frameSet) {
        Dataset<String, String> dataset = new Dataset<String, String>();
        for (Token predicate : frameSet.getPredicateList()) {
            for (Token argument : argumentCandidates(frameSet, predicate)) {
                BasicDatum<String, String> datum =
                        (BasicDatum<String, String>) featureGenerator.datumFrom(frameSet, argument, predicate);
                String label = NIL_LABEL;
                for (Map.Entry<Token, String> p : frameSet.argumentsOf(predicate).entrySet()) {
                    if (p.getKey().equals(argument)) {
                        label = p.getValue();
                        break;
                    }
                }
                datum.setLabel(label);
                dataset.add(datum);
            }
        }

        return dataset;
    }

    public Dataset<String, String> datasetFrom(Collection<SemanticFrameSet> frameSets) {
        Dataset<String, String> dataset = new Dataset<String, String>();
        for (SemanticFrameSet frameSet : frameSets)
            dataset.addAll(datasetFrom(frameSet));

        dataset.applyFeatureCountThreshold(3);

        return dataset;
    }


    public void update(SemanticFrameSet predictedFrame, SemanticFrameSet goldFrame) {
        Dataset<String, String> dataset = new Dataset<String, String>();

        for (Token predicate : goldFrame.getPredicateList()) {

            if (predictedFrame.isPredicate(predicate)) {
                Map<Token, String> goldArguments = goldFrame.argumentsOf(predicate);
                Map<Token, String> predictedArguments = predictedFrame.argumentsOf(predicate);

                for (Token t : argumentCandidates(predictedFrame, predicate)) {
                    String goldLabel = goldArguments.get(t);
                    String predictedLabel = predictedArguments.get(t);

                    if (goldLabel == null || goldLabel.equals(ArgumentClassifier.NIL_LABEL))
                        goldLabel = PerceptronClassifier.PREDICTED_LABEL_PREFIX + ArgumentClassifier.NIL_LABEL;
                    else
                        goldLabel = PerceptronClassifier.PREDICTED_LABEL_PREFIX + goldLabel;

                    if (predictedLabel == null || predictedLabel.equals(ArgumentClassifier.NIL_LABEL))
                        predictedLabel = PerceptronClassifier.PREDICTED_LABEL_PREFIX + ArgumentClassifier.NIL_LABEL;
                    else
                        predictedLabel = PerceptronClassifier.PREDICTED_LABEL_PREFIX + predictedLabel;

                    BasicDatum<String, String> datum =
                            (BasicDatum<String, String>) featureGenerator.datumFrom(predictedFrame, t, predicate);

                    datum.addLabel(goldLabel);
                    datum.addLabel(predictedLabel);

                    dataset.add(datum);

                }

            }

            //TODO: what if token is predicate in one but not other
        }

        classifier.manualTrain(dataset);

    }

    public ArgumentFeatureGenerator getFeatureGenerator() {
        return featureGenerator;
    }
}
