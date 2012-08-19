package spinach.predicateclassifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.BasicDatum;
import spinach.classify.Classifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;

import java.util.HashSet;
import java.util.List;

public class PredicateClassifier {

    protected final Classifier classifier;
    protected final PredicateFeatureGenerator featureGenerator;

    public PredicateClassifier(Classifier classifier, PredicateFeatureGenerator featureGenerator){
        this.classifier = classifier;
        this.featureGenerator = featureGenerator;
    }

    public SemanticFrameSet predicatesOf(SemanticFrameSet sentenceFrame){
        return predicatesOf(sentenceFrame.sentence());
    }

    protected SemanticFrameSet predicatesOf(TokenSentence sentence){
        SemanticFrameSet frameSet = new SemanticFrameSet(sentence);
        for (Token t : sentence){
            if (classifier.classOf(featureGenerator.datumFrom(frameSet, t)).equals("predicate"))
                frameSet.addPredicate(t);
        }

        return frameSet;
    }

    public Dataset<String, String> goldDataset(SemanticFrameSet goldFrames){
        Dataset<String, String> dataset = new Dataset<String, String>();
        HashSet<Token> predicates = new HashSet<Token>(goldFrames.getPredicateList());
        for (Token t : goldFrames.sentence()){
            BasicDatum<String, String> datum = (BasicDatum<String, String>) featureGenerator.datumFrom(goldFrames, t);
            if (predicates.contains(t))
                datum.setLabel("predicate");
            else
                datum.setLabel("not_predicate");
            dataset.add(datum);
        }

        return dataset;
    }

    public Dataset<String, String> goldDataset(List<SemanticFrameSet> goldFrameSet){
        Dataset<String, String> dataset = new Dataset<String, String>();
        for (SemanticFrameSet frameSet : goldFrameSet)
            dataset.addAll(goldDataset(frameSet));

        return dataset;
    }

}
