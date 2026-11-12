import SwiftUI

/// The personal dictionary screen: the words Phonon should spell the user's way (a term, or "phrase → replacement"),
/// add one with an optional "Write instead", remove one. Stored by Session in dictionary.json (the desktop schema).
/// Takes plain values and callbacks, so the unit tests render it offscreen without a Session.
struct DictionaryView: View {
    let entries: [UserDictionary.Entry]
    var onAdd: (String, String?) -> Void
    var onRemove: (UserDictionary.Entry) -> Void
    var onClose: (() -> Void)?

    var body: some View {
        ScrollView {
            DictionaryContent(entries: entries, onAdd: onAdd, onRemove: onRemove, onClose: onClose)
        }
        .scrollIndicators(.hidden)
        .scrollDismissesKeyboard(.interactively)
        .background(Ember.window.ignoresSafeArea())
    }
}

/// The screen without its scroll view (what the offscreen renders draw).
struct DictionaryContent: View {
    let entries: [UserDictionary.Entry]
    var onAdd: (String, String?) -> Void
    var onRemove: (UserDictionary.Entry) -> Void
    var onClose: (() -> Void)?
    /// offscreen renders: the add form's fields drawn as they look, since UIKit-backed text fields do not render there
    var staticFields = false
    @State private var phrase = ""
    @State private var replacement = ""
    @FocusState private var focus: Field?

    private enum Field { case phrase, replacement }

    init(entries: [UserDictionary.Entry], onAdd: @escaping (String, String?) -> Void,
         onRemove: @escaping (UserDictionary.Entry) -> Void, onClose: (() -> Void)? = nil,
         staticFields: Bool = false, phrase: String = "", replacement: String = "") {
        self.entries = entries
        self.onAdd = onAdd
        self.onRemove = onRemove
        self.onClose = onClose
        self.staticFields = staticFields
        _phrase = State(initialValue: phrase)
        _replacement = State(initialValue: replacement)
    }

    private var canAdd: Bool { !phrase.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            addCard
            listCard
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
        .padding(.bottom, 32)
        .foregroundStyle(Ember.ink)
        .background(Ember.window)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Dictionary").font(.system(size: 30, weight: .semibold))
                Spacer()
                if let onClose {
                    Button("Done", action: onClose)
                        .font(.system(size: 16, weight: .semibold))
                        .buttonStyle(.plain)
                        .foregroundStyle(Ember.ember2)
                }
            }
            Text("Names, products and jargon you say often. Phonon spells them your way. The list stays on this iPhone.")
                .font(.system(size: 13)).foregroundStyle(Ember.mute)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, 8)
    }

    // MARK: add

    private var addCard: some View {
        Card(title: "Add a word") {
            field("Word or phrase", text: $phrase, which: .phrase)
            field("Write instead (optional)", text: $replacement, which: .replacement)
            Text("Leave “Write instead” empty for a word to spell right, like Kubernetes. Fill it in to swap a phrase, like vee llm → vLLM.")
                .font(.system(size: 12)).foregroundStyle(Ember.mute)
                .fixedSize(horizontal: false, vertical: true)
            Button(action: add) {
                Text("Add")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
                    .background(RoundedRectangle(cornerRadius: 12).fill(canAdd ? Ember.ember : Ember.panelRaised))
                    .foregroundStyle(canAdd ? Color.white : Ember.dim)
            }
            .buttonStyle(.plain)
            .disabled(!canAdd)
            .padding(.top, 2)
        }
    }

    @ViewBuilder
    private func field(_ prompt: String, text: Binding<String>, which: Field) -> some View {
        Group {
            if staticFields {
                Text(text.wrappedValue.isEmpty ? prompt : text.wrappedValue)
                    .foregroundStyle(text.wrappedValue.isEmpty ? Ember.dim : Ember.ink)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                TextField("", text: text, prompt: Text(prompt).foregroundStyle(Ember.dim))
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .submitLabel(which == .phrase ? .next : .done)
                    .focused($focus, equals: which)
                    .onSubmit { if which == .phrase { focus = .replacement } else { add() } }
                    .foregroundStyle(Ember.ink)
                    .tint(Ember.ember)
            }
        }
        .font(.system(size: 15))
        .padding(.horizontal, 12)
        .frame(height: 44)
        .background(RoundedRectangle(cornerRadius: 10).fill(Ember.panelRaised))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(focus == which ? Ember.ember.opacity(0.6) : Ember.line, lineWidth: 1))
    }

    private func add() {
        guard canAdd else { return }
        let rep = replacement.trimmingCharacters(in: .whitespacesAndNewlines)
        onAdd(phrase, rep.isEmpty ? nil : rep)
        phrase = ""
        replacement = ""
        focus = .phrase
    }

    // MARK: list

    private var listCard: some View {
        Card(title: entries.isEmpty ? "Your words" : "Your words · \(entries.count)") {
            if entries.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Image(systemName: "character.book.closed")
                        .font(.system(size: 22)).foregroundStyle(Ember.ember)
                    Text("No words yet").font(.system(size: 15, weight: .medium))
                    Text("Add the words Phonon gets wrong. They are used for every dictation, in every app.")
                        .font(.system(size: 13)).foregroundStyle(Ember.mute)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, 4)
            } else {
                ForEach(Array(entries.enumerated()), id: \.element) { i, e in
                    row(e)
                    if i < entries.count - 1 { Divider().overlay(Ember.line) }
                }
            }
        }
    }

    private func row(_ e: UserDictionary.Entry) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                if let rep = e.replacement {
                    HStack(spacing: 6) {
                        Text(e.phrase).foregroundStyle(Ember.soft)
                        Image(systemName: "arrow.right").font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(Ember.mute)
                        Text(rep).foregroundStyle(Ember.ink)
                    }
                    .font(.system(size: 15, weight: .medium))
                } else {
                    Text(e.phrase).font(.system(size: 15, weight: .medium))
                }
                if !e.spokenForms.isEmpty {
                    Text("also heard as " + e.spokenForms.joined(separator: ", "))
                        .font(.system(size: 12)).foregroundStyle(Ember.mute)
                }
            }
            .lineLimit(2)
            Spacer(minLength: 0)
            Button {
                onRemove(e)
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(Ember.mute)
                    .frame(width: 36, height: 36)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Remove \(e.replacement ?? e.phrase)")
        }
        .padding(.vertical, 4)
    }
}

/// A rounded panel with an optional small-caps title: the app's card.
struct Card<Content: View>: View {
    var title: String?
    @ViewBuilder var content: Content

    init(title: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let title {
                Text(title.uppercased())
                    .font(.system(size: 11, weight: .semibold))
                    .tracking(0.8)
                    .foregroundStyle(Ember.mute)
            }
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 16).fill(Ember.panel))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Ember.line, lineWidth: 1))
    }
}
