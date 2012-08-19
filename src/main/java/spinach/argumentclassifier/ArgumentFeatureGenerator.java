package spinach.argumentclassifier;

import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.*;

public class ArgumentFeatureGenerator {

    public Datum<String, String> datumFrom(SemanticFrameSet frameSet,
                                           Token argument, Token predicate){
        return new BasicDatum<String, String>(featuresOf(frameSet, argument, predicate));
    }

    public Collection<String> featuresOf(TokenSentenceAndPredicates sentenceAndPredicates,
                                  Token argument, Token predicate){

        Collection<String> features = new ArrayList<String>();

        /*
		 * Feature 1: argument (and modifier), predicate split lemma, form; pposs
		 */
        features.add("argsplm|" + argument.lemma);
        features.add("argspfm|" + argument.form);
        features.add("argppos|" + argument.pos);
        features.add("predsplm|" + predicate.lemma);
        features.add("predspfm|" + predicate.form);
        features.add("predppos|" + predicate.pos);

        Token pmod = getPMOD(sentenceAndPredicates, argument);
        if (pmod != null){
            features.add("pmodsplm|" + pmod.lemma);
            features.add("pmodspfm|" + pmod.form);
            features.add("pmodppos|" + pmod.pos);
        }


        /*
           * Feature 2: pos/deprel for predicate children, children of predicate ancestor across VC/IM dependencies
           */
        StringBuilder relationFeature = new StringBuilder("predcdeprel|");
        StringBuilder posFeature = new StringBuilder("predcpposs|");
        for (Token child : sentenceAndPredicates.getChildren(predicate)){
            relationFeature.append(child.syntacticHeadRelation).append(" ");
            posFeature.append(child.pos).append(" ");
        }
        features.add(relationFeature.toString());
        features.add(posFeature.toString());

        relationFeature = new StringBuilder("vcimdeprel|");
        posFeature = new StringBuilder("vcimpposs|");
        Token vcimAncestor = predicate;
        while(vcimAncestor.syntacticHeadRelation.equals("VC") || vcimAncestor.syntacticHeadRelation.equals("IM"))
            vcimAncestor = sentenceAndPredicates.getParent(vcimAncestor);

        for (Token ancestorChild : sentenceAndPredicates.getChildren(vcimAncestor)){
            relationFeature.append(" ").append(ancestorChild.syntacticHeadRelation);
            posFeature.append(" ").append(ancestorChild.pos);
            if (ancestorChild.equals(argument)){
                relationFeature.append("a");
                posFeature.append("a");
            }
            else if (ancestorChild.equals(predicate)){
                relationFeature.append("p");
                posFeature.append("p");
            }
        }
        features.add(relationFeature.toString());
        features.add(posFeature.toString());


        /*
           * Feature 3: dependency path
           */
        StringBuilder path = new StringBuilder();

        /*
           * Ancestor splits the path into two halves:
           * from the argument to the ancestor, the dependencies go upwards;
           * from the predicate to the ancestor, the dependencies go downwards
           */
        Token ancestor = sentenceAndPredicates.getCommonAncestor(argument, predicate);

        Deque<Token> argPath = sentenceAndPredicates.ancestorPath(argument, ancestor);
        Deque<Token> predPath = sentenceAndPredicates.ancestorPath(predicate, ancestor);

        argPath.removeLast(); //ancestor is last thing in both pathA and pathB, don't need it
        predPath.removeLast();

        while(!argPath.isEmpty()){	//argPath is (upward) path from argument to ancestor
            path.append(argPath.removeFirst().syntacticHeadRelation).append("^ ");
        }
        while(!predPath.isEmpty()){	//predPath is (downwards) path from predicate to ancestor
            path.append(predPath.removeLast().syntacticHeadRelation).append("v ");
        }

        features.add("path|" + path.toString());
        features.add("pathpos|" + argument.pos + " " + path.toString() + predicate.pos);	//with poss tags
        features.add("pathlem|" + argument.lemma + " " + path.toString() + predicate.lemma);	//with splm tagss


        return features;
    }

    private static Token getPMOD(TokenSentence sentence, Token argument){
        for (Token t : sentence.getChildren(argument))
            if (t.syntacticHeadRelation.startsWith("PMOD"))
                return t;
        return null;
    }

}
