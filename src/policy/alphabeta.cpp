#include <utility>
#include "state.hpp"
#include "alphabeta.hpp"


/*============================================================
 * AlphaBeta — eval_ctx
 *
 * Negamax with pruning. Caller manages memory.
 *============================================================*/
int AlphaBeta::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // return the score for a winning terminal state

    if(state->game_state ==  WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        ); 
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop === */
    int best_score = M_MAX;

    for(auto& action : state->legal_actions){
        // create the child state after applying action

        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();

        // search the child one level deeper
        int raw_score;
        // convert raw to the current player's perspective.
        int score;
        if(same){
            raw_score = eval_ctx(next, depth -1, alpha, beta, history, ply + 1, ctx, p);
            score = raw_score;
        } else {
            raw_score = eval_ctx(next, depth -1, -beta, -alpha, history, ply + 1, ctx, p);
            score = -raw_score;
        }
        delete next;

        // update best_score if this child is better.
        if(score > best_score){
            best_score = score;
        }
        if(best_score > alpha){
            alpha = best_score;
        }

        if(alpha >= beta){
            break;
        }

    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult AlphaBeta::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }


    int best_score = M_MAX - 10;
    int alpha = M_MAX;
    int beta = P_MAX;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    for(auto& action : state->legal_actions){
        /* [ Hackathon TODO 4-1 ]
         * search this move like TODO 3, but starting from the root */

        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();
        int raw_score;
        int score;
        if(same){
            raw_score = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
            score = raw_score;
        } else {
            raw_score = eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
            score = -raw_score;
        }
        delete next;

        if(score > best_score){
            // keep this move if it is the best so far

            best_score = score;
            alpha = best_score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }  
        move_index++;
    }

    // update result and return
    result.score = best_score;
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;
    result.depth = depth;
    result.pv.clear();
    if(result.best_move != Move()){
        result.pv.push_back(result.best_move);
    }
    return result;
} 


/*============================================================
 * AlphaBeta — default_params / param_defs
 *============================================================*/
ParamMap AlphaBeta::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> AlphaBeta::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
