//! Lower typed seeds into `Seeds.run`, sharing the body with `db/seed.sql`.
//!
//! Association creates become explicit model creates with foreign keys.
//! Script constants become locals: Ruby rejects constant assignments inside
//! methods. Roundhouse models these constants as private to the seed script.

use crate::App;
use crate::dialect::LibraryFunction;
use crate::effect::EffectSet;
use crate::expr::{Expr, ExprNode, LValue, Literal};
use crate::ident::{Symbol, VarId};
use crate::lower::typing::fn_sig;
use crate::span::Span;
use crate::ty::Ty;

/// Build the `Seeds` module as a single LibraryFunction:
/// `Seeds.run() -> nil` carrying the lowered seeds body.
/// Empty when the app has no seeds file.
pub fn lower_seeds_to_library_functions(app: &App) -> Vec<LibraryFunction> {
    let Some(body) = app.seeds.as_ref().cloned() else {
        return Vec::new();
    };
    let body = lower_seeds_body(&body);
    let module_path = vec![Symbol::from("Seeds")];
    vec![LibraryFunction {
        module_path,
        name: Symbol::from("run"),
        params: Vec::new(),
        unsupported_formals: None,
        has_anonymous_block: false,
        body,
        signature: Some(fn_sig(vec![], Ty::Nil)),
        effects: EffectSet::default(),
        is_async: false,
    }]
}

/// The seeds body every consumer works from: `Seeds.run` and the
/// static `db/seed.sql` evaluate the same lowered form.
pub fn lower_seeds_body(seeds: &Expr) -> Expr {
    seed_constants_to_locals(&rewrite_assoc_create(seeds))
}

/// Collect unconditional script constant writes in source order.
/// Conditional definitions stay unsupported: a skipped local assignment
/// would read `nil` where an undefined Ruby constant raises NameError.
pub fn seed_constant_writes(seeds: &Expr) -> Vec<(&Symbol, &Expr)> {
    fn walk<'a>(e: &'a Expr, out: &mut Vec<(&'a Symbol, &'a Expr)>) {
        match &*e.node {
            ExprNode::Seq { exprs } => exprs.iter().for_each(|e| walk(e, out)),
            // `return if COND` + rest ingests to a synthetic
            // `if COND then nil else rest end` (`detect_leading_guard`).
            ExprNode::If { then_branch, else_branch, .. }
                if e.span.is_synthetic()
                    && matches!(&*then_branch.node, ExprNode::Lit { value: Literal::Nil }) =>
            {
                walk(else_branch, out);
            }
            ExprNode::Assign { target: LValue::Const { path }, value } if path.len() == 1 => {
                out.push((&path[0], value));
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(seeds, &mut out);
    out
}

/// Rewrite script constants to distinct locals, preserving analyzed types.
fn seed_constants_to_locals(seeds: &Expr) -> Expr {
    let consts = seed_constant_writes(seeds);
    if consts.is_empty() {
        return seeds.clone();
    }
    let mut used = std::collections::HashSet::new();
    collect_local_names(seeds, &mut used);
    let mut locals = std::collections::HashMap::new();
    // Allocate in source order, reserving generated names as well as source locals.
    for (name, _) in consts {
        if locals.contains_key(name) {
            continue;
        }
        let mut local = crate::naming::snake_case(name.as_str());
        while !used.insert(local.clone()) {
            local.push_str("_const");
        }
        locals.insert(name.clone(), Symbol::from(local));
    }
    fn rewrite(e: &mut Expr, locals: &std::collections::HashMap<Symbol, Symbol>) {
        let replacement = match &mut *e.node {
            ExprNode::Const { path } if path.len() == 1 && locals.contains_key(&path[0]) => {
                Some(ExprNode::Var { id: VarId(0), name: locals[&path[0]].clone() })
            }
            ExprNode::Assign { target, .. } => {
                if let LValue::Const { path } = target {
                    if path.len() == 1 && locals.contains_key(&path[0]) {
                        *target = LValue::Var { id: VarId(0), name: locals[&path[0]].clone() };
                    }
                }
                None
            }
            _ => None,
        };
        if let Some(node) = replacement {
            *e.node = node;
        }
        e.node.for_each_child_mut(&mut |c| rewrite(c, locals));
    }
    let mut out = seeds.clone();
    rewrite(&mut out, &locals);
    out
}

fn collect_local_names(e: &Expr, out: &mut std::collections::HashSet<String>) {
    match &*e.node {
        ExprNode::Var { name, .. } | ExprNode::Assign { target: LValue::Var { name, .. }, .. } => {
            out.insert(name.as_str().to_string());
        }
        ExprNode::Lambda { params, .. } => {
            out.extend(params.iter().map(|p| p.as_str().to_string()));
        }
        _ => {}
    }
    e.node.for_each_child(&mut |c| collect_local_names(c, out));
}

/// `<parent>.<assoc>.create(<args>)` → `<AssocClass>.create(<parent_class>_id: <parent>.id, ...args)`.
///
/// Pattern shape (after typer):
///   Send {                                     ← outer (create/create!)
///     recv: Some(Send {                        ← inner (assoc reader)
///       recv: Some(parent_expr),               ← typed Ty::Class { id: <Parent> }
///       method: <assoc> (e.g. "comments"),
///       args: [],
///     }),
///     method: "create" | "create!",
///     args: [Hash{...}] | [],
///   }
///
/// Transforms to:
///   Send {
///     recv: Some(Const(<AssocClass>)),
///     method: "create" | "create!",
///     args: [Hash{ <parent_class>_id: parent_expr.id, ...original_entries }],
///   }
///
/// `<AssocClass>` is the association's DECLARED target when the parent
/// model is known — `has_many :push_subscriptions, class_name:
/// "Push::Subscription"` names a class the assoc name alone never
/// reaches (campfire's test built a `PushSubscription`, an
/// uninitialized constant) — and singularize+camelize on the assoc
/// method name otherwise. `<parent_class>_id` derives from the
/// parent_expr's type (a Class type the typer set), snake_cased — e.g.
/// `Article` → `article_id`. Falls through (returns the expr
/// unchanged) when any of the shape preconditions don't hold.
pub fn rewrite_assoc_create(expr: &Expr) -> Expr {
    rewrite_assoc_create_with_models(expr, &[])
}

pub fn rewrite_assoc_create_with_models(expr: &Expr, models: &[crate::dialect::Model]) -> Expr {
    crate::lower::controller_to_library::util::map_expr(expr, &|e| assoc_create_rewrite(e, models))
}

/// In-place twin. Returns whether any node changed so the test
/// lowerer can skip a follow-up typing pass.
pub fn rewrite_assoc_create_with_models_in_place(
    expr: &mut Expr,
    models: &[crate::dialect::Model],
) -> bool {
    crate::lower::controller_to_library::util::map_expr_mut(expr, &|e| {
        assoc_create_rewrite(e, models)
    })
}

fn assoc_create_rewrite(e: &Expr, models: &[crate::dialect::Model]) -> Option<Expr> {
    let declared_target = |parent: &str, assoc: &str| -> Option<String> {
        let model = models.iter().find(|m| m.name.0.as_str() == parent)?;
        model.associations().find_map(|a| match a {
            crate::dialect::Association::HasMany { name, target, .. }
                if name.as_str() == assoc =>
            {
                Some(target.0.as_str().to_string())
            }
            _ => None,
        })
    };
    let ExprNode::Send {
        recv: Some(outer_recv),
        method: outer_method,
        args: outer_args,
        block: None,
        ..
    } = &*e.node
    else {
        return None;
    };
    let outer_method_str = outer_method.as_str();
    // `create` / `create!` map to the same-named class methods.
    // `build` is Rails-specific (instantiate without saving) —
    // map to `new` so Ruby/CRuby callers get `<Class>.new(attrs)`
    // and the TS emitter renders that as `new <Class>(attrs)`.
    let target_method = match outer_method_str {
        "create" | "create!" => outer_method.clone(),
        "build" => Symbol::from("new"),
        _ => return None,
    };
    let ExprNode::Send {
        recv: Some(parent_expr),
        method: assoc_method,
        args: inner_args,
        block: None,
        ..
    } = &*outer_recv.node
    else {
        return None;
    };
    if !inner_args.is_empty() {
        return None;
    }
    // Resolve parent class from the typer-set type. Without the
    // class type we can't derive the FK name; fall through.
    let parent_class = match parent_expr.ty.as_ref() {
        Some(Ty::Class { id, .. }) => id.0.as_str().to_string(),
        _ => return None,
    };
    let assoc_class = declared_target(&parent_class, assoc_method.as_str())
        .unwrap_or_else(|| crate::naming::singularize_camelize(assoc_method.as_str()));
    let fk = format!("{}_id", crate::naming::snake_case(&parent_class));

    // Build the FK-id expr: `<parent_expr>.id`.
    let parent_id = Expr::new(
        parent_expr.span,
        ExprNode::Send {
            recv: Some(parent_expr.clone()),
            method: Symbol::from("id"),
            args: vec![],
            block: None,
            parenthesized: false,
        },
    );
    let fk_entry = (
        Expr::new(
            e.span,
            ExprNode::Lit { value: crate::expr::Literal::Sym { value: Symbol::from(fk) } },
        ),
        parent_id,
    );

    // Merge FK entry with original kwargs/hash. Real-blog's
    // `article.comments.create!(commenter:, body:)` parses the
    // trailing kwargs as a single Hash arg.
    let merged_hash = match outer_args.first().map(|a| (&*a.node, a.span)) {
        Some((ExprNode::Hash { entries, kwargs }, span)) => {
            let mut new_entries = vec![fk_entry];
            new_entries.extend(entries.iter().cloned());
            Expr::new(span, ExprNode::Hash { entries: new_entries, kwargs: *kwargs })
        }
        _ => Expr::new(
            e.span,
            ExprNode::Hash { entries: vec![fk_entry], kwargs: true },
        ),
    };

    Some(Expr::new(
        e.span,
        ExprNode::Send {
            recv: Some(Expr::new(
                e.span,
                ExprNode::Const {
                    path: assoc_class.split("::").map(Symbol::from).collect(),
                },
            )),
            method: target_method,
            args: vec![merged_hash],
            block: None,
            parenthesized: true,
        },
    ))
}

// `Span` used in synthesized expressions above.
#[allow(unused_imports)]
use Span as _Span;
