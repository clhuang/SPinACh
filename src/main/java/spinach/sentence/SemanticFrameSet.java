package spinach.sentence;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import edu.stanford.nlp.util.Pair;

import java.util.Collection;

/**
 * @author  Calvin Huang
 *
 * A SemanticFrameSet represents relationships between tokens
 * in a single sentence. Each SemanticFrameSet contains a sentence,
 * containing tokens and their syntactic relationships, and it also
 * contains a list of predicates, and mappings from those predicates
 * to their arguments, and the semantic relations.
 */
public class SemanticFrameSet extends TokenSentenceAndPredicates{

    public SemanticFrameSet(){}

    public SemanticFrameSet(TokenSentenceAndPredicates sentenceAndPredicates){
        super(sentenceAndPredicates);
        predicateList = sentenceAndPredicates.predicateList;
    }

    private Multimap<Token, Pair<Token, String>> relations =
            HashMultimap.create();

    /**
     * Adds a relation between an argument and a predicate.
     *
     * @param predicate predicate being referenced
     * @param argument argument being referenced
     * @param relation semantic relationship between the two
     */
    public void addArgument(Token predicate, Token argument, String relation){
        relations.put(predicate, new Pair<Token, String>(argument, relation));
    }

    /**
     * View the relations and arguments of some predicate
     *
     * @param predicate predicate to look at the arguments of
     * @return arguments of this predicate, and their relations
     */
    public Collection<Pair<Token, String>> argumentsOf(Token predicate){
        return relations.get(predicate);
    }

}
