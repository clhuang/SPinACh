package spinach.sentence;

import java.util.Collections;
import java.util.HashMap;
import java.util.ListIterator;
import java.util.Map;

/**
 *
 * A SemanticFrameSet represents relationships between tokens
 * in a single sentence. Each SemanticFrameSet contains a sentence,
 * containing tokens and their syntactic relationships, and it also
 * contains a list of predicates, and mappings from those predicates
 * to their arguments, and the semantic relations.
 *
 * @author Calvin Huang
 *
 */
public class SemanticFrameSet extends TokenSentenceAndPredicates {

    /*private Multimap<Token, Pair<Token, String>> relations =
            HashMultimap.create();*/

    private Map<Token, Map<Token, String>> relations = new HashMap<Token, Map<Token, String>>();

    public SemanticFrameSet() {
    }

    public SemanticFrameSet(TokenSentenceAndPredicates sentenceAndPredicates) {
        super(sentenceAndPredicates);
        predicateList = sentenceAndPredicates.predicateList;
    }

    /**
     * Adds a relation between an argument and a predicate.
     *
     * @param predicate predicate being referenced
     * @param argument  argument being referenced
     * @param relation  semantic relationship between the two
     */
    public void addArgument(Token predicate, Token argument, String relation) {
        //relations.put(predicate, new Pair<Token, String>(argument, relation));
        Map<Token, String> predicateMap = argumentsOf(predicate);
        if (predicateMap == null){
            predicateMap = new HashMap<Token, String>();
            predicateMap.put(argument, relation);
            relations.put(predicate, predicateMap);
        }
        else
            predicateMap.put(argument, relation);

    }

    /**
     * View the relations and arguments of some predicate
     *
     * @param predicate predicate to look at the arguments of
     * @return arguments of this predicate, and their relations
     */
    public Map<Token, String> argumentsOf(Token predicate) {
        return Collections.unmodifiableMap(relations.get(predicate));
    }

    /**
     * Trim the list of predicates--any predicate without arguments is removed
     */
    public void trimPredicates() {
        for (ListIterator<Token> iterator =
                     predicateList.listIterator(predicateList.size());
             iterator.hasPrevious(); )
            if (argumentsOf(iterator.previous()).isEmpty())
                iterator.remove();
    }

}
