#include <utility>
#include "state.hpp"
#include "quiescence.hpp"
#include <algorithm>
#include "config.hpp"

static int move_order_score(State* state, const Move& move){
    Point from = move.first;
    Point to = move.second;

    int self = state->player;
    int opp = 1 - self;

    int attacker = state->board.board[self][from.first][from.second];
    int victim = state->board.board[opp][to.first][to.second];

    int score = 0;

    if(victim){
        score += 10000;
        score += PIECE_VALUES[victim] * 10;
        score -= PIECE_VALUES[attacker];

        if(victim == 6){
            score += 100000;
        }
    }

    //Promotion bonus
    if(attacker == 1 && (to.first == 0 || to.first == BOARD_H - 1)){
        score += 5000;
    }

    int center_r2 = 2 * (int)to.first - (BOARD_H - 1);
    int center_c2 = 2 * (int)to.second - (BOARD_W - 1);
    score -= center_r2 * center_r2 + center_c2 * center_c2;

    return score;
}

static std::vector<Move> ordered_moves(State* state){
    std::vector<Move> moves = state->legal_actions;
    
    std::sort(moves.begin(), moves.end(),
        [&](const Move& a, const Move& b){
            return move_order_score(state, a) > move_order_score(state, b);
        }
    );
    return moves;
}

static bool is_capture_move(State* state, const Move& move){
    Point to = move.second;
    int opp = 1 - state->player;
    return state->board.board[opp][to.first][to.second] != 0;
}

static std::vector<Move> capture_moves(State* state){
    std::vector<Move> moves;

    for(const auto& move: state->legal_actions){
        if(is_capture_move(state, move)){
            moves.push_back(move);
        }
    }

    std::sort(moves.begin(), moves.end(),
        [&](const Move& a, const Move& b){
            return move_order_score(state, a) > move_order_score(state, b);
        }
    );

    return moves;
}

static int quiescence(
    State* state,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int qdepth
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }

    if(ctx.stop){
        return 0;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    int stand_pat = state->evaluate(
        p.use_kp_eval, p.use_eval_mobility, &history
    );

    if(stand_pat >= beta){
        return beta;
    }

    if(stand_pat > alpha){
        alpha = stand_pat;
    }

    if(qdepth <= 0){
        return alpha;
    }

    auto moves = capture_moves(state);

    for(auto& action : moves){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int raw_score;
        int score;

        if(same){
            raw_score = quiescence(
                next, alpha, beta, history, ply + 1, ctx, p, qdepth - 1
            );
            score = raw_score;
        } else {
            raw_score = quiescence(
                next, -beta, -alpha, history, ply + 1, ctx, p, qdepth - 1
            );
            score = -raw_score;
        }

        delete next;

        if(score >= beta){
            return beta;
        }

        if(score > alpha){
            alpha = score;
        }
    }
    return alpha;
    
}

/*============================================================
 * Quiescence — eval_ctx
 *
 * Negamax with pruning. Caller manages memory.
 *============================================================*/
int Quiescence::eval_ctx(
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
        int score = quiescence(
        state, alpha, beta, history, ply, ctx, p, 4
        );
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop === */
    int best_score = M_MAX;
    bool first_child = true;
    
    auto moves = ordered_moves(state);

    for(auto& action : moves){
        // create the child state after applying action

        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();

        // search the child one level deeper
        int raw_score;
        // convert raw to the current player's perspective.
        int score;
        if(first_child){

            //First move: full window search
            if(same){
                raw_score = eval_ctx(next, depth -1, alpha, beta, history, ply + 1, ctx, p);
                score = raw_score;
            } else {
                raw_score = eval_ctx(next, depth -1, -beta, -alpha, history, ply + 1, ctx, p);
                score = -raw_score;
            }
            first_child = false;
        } else {
            // Other moves: null-window search
            if(same){
                raw_score = eval_ctx(next, depth -1, alpha, alpha + 1, history, ply + 1, ctx, p);
                score = raw_score;
            } else {
                raw_score = eval_ctx(next, depth -1, -alpha -1, -alpha, history, ply + 1, ctx, p);
                score = -raw_score;
            }

            //If is maybe better, re-search with full window
            if(score > alpha && score < beta){
                if(same){
                    raw_score = eval_ctx(next, depth -1, alpha, beta, history, ply + 1, ctx, p);
                    score = raw_score;
                } else {
                    raw_score = eval_ctx(next, depth -1, -beta, -alpha, history, ply + 1, ctx, p);
                    score = -raw_score;
                }
            }
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
 * Quiescence — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult Quiescence::search(
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
    bool first_child = true;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    auto moves = ordered_moves(state);

    for(auto& action : moves){
        /* [ Hackathon TODO 4-1 ]
         * search this move like TODO 3, but starting from the root */

        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();
        int raw_score;
        int score;
        if(first_child){

            //First move: full window search
            if(same){
                raw_score = eval_ctx(next, depth -1, alpha, beta, history, 1, ctx, p);
                score = raw_score;
            } else {
                raw_score = eval_ctx(next, depth -1, -beta, -alpha, history, 1, ctx, p);
                score = -raw_score;
            }
            first_child = false;
        } else {
            // Other moves: null-window search
            if(same){
                raw_score = eval_ctx(next, depth -1, alpha, alpha + 1, history, 1, ctx, p);
                score = raw_score;
            } else {
                raw_score = eval_ctx(next, depth -1, -alpha -1, -alpha, history, 1, ctx, p);
                score = -raw_score;
            }

            //If is maybe better, re-search with full window
            if(score > alpha && score < beta){
                if(same){
                    raw_score = eval_ctx(next, depth -1, alpha, beta, history, 1, ctx, p);
                    score = raw_score;
                } else {
                    raw_score = eval_ctx(next, depth -1, -beta, -alpha, history, 1, ctx, p);
                    score = -raw_score;
                }
            }
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
 * Quiescene — default_params / param_defs
 *============================================================*/
ParamMap Quiescence::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> Quiescence::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
