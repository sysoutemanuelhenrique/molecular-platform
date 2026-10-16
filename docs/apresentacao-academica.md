# Turing Bohr Molecular Platform — Apresentação Acadêmica

## O que é

A Turing Bohr Molecular Platform é uma plataforma de pesquisa, em estágio
inicial, para registrar e rastrear objetos moleculares — sua proveniência
de aquisição, sua identidade científica e as transformações entre estados
— por meio de um registro auditável em arquivo: um log de atividade por
experimento, sempre por acréscimo (append-only), e registros científicos
(insumos, entidades, estados, conformers, condições declaradas) que são
create-once — nunca sobrescritos depois de persistidos. A exceção
deliberada são os registros de uma *tentativa* de operação em andamento
(uma promoção ou uma execução): seu próprio status é atualizado conforme a
operação avança, até alcançar um desfecho terminal — que, esse sim, nunca é
reaberto.

O desenvolvimento segue um processo deliberadamente lento: cada incremento
de funcionalidade é especificado por escrito antes de qualquer código ser
escrito, revisado adversarialmente por uma sessão independente antes de
ser aceito, e validado por uma suíte de testes que, nesta data, soma mais
de 1.400 casos. Esta página descreve exclusivamente o que já está
implementado e testado — não uma visão de futuro, e não uma promessa.

## O que já está implementado

- **Registro de entradas (ligantes e proteínas)** com proveniência completa
  de aquisição (método, origem, checksum) — incluindo download direto do
  PubChem por CID.
- **Identidade científica**: `ScientificEntity` (identidade estável de um
  objeto molecular), `MolecularState` (uma realização química explícita
  dessa identidade) e `Conformer` (uma geometria específica de um estado) —
  três conceitos distintos, nunca confundidos entre si nem com o arquivo
  bruto de origem.
- **Promoção de ligante**: o caminho testado e auditado de um arquivo de
  ligante registrado até seu primeiro par Entity/State, usando RDKit (versão
  fixada) para parsing e sanitização, com verificação de checksum no momento
  da promoção.
- **Rastreamento de proveniência de transformações de estado** — incluindo
  um registro específico (`state derive`) que **não executa** protonação ou
  tautomerização; apenas registra que uma transformação já realizada por
  outro meio ocorreu, e como.
- **Enumeração de tautômeros** via RDKit, como capacidade separada: enumera
  candidatos mecanicamente, sem decidir qual é o "correto" biologicamente.
- **Declarações de condições experimentais** (pH) como registro imutável,
  nunca inferido.
- **Fundação de preparação de receptor** (inspeção, seleção e resolução de
  um arquivo de estrutura mmCIF/PDB legado via Gemmi) — "read-only" aqui
  descreve especificamente o arquivo de estrutura original: ele nunca é
  modificado, e nenhuma etapa desta fundação grava um novo arquivo de
  coordenadas preparado a partir dele. O próprio processo de inspeção,
  seleção e resolução **persiste seus próprios registros** (relatórios de
  inspeção, políticas de seleção declaradas, resoluções) — inspecionar,
  selecionar e resolver não são operações que não escrevem nada. Nunca
  atribui hidrogênios ou estado de protonação, nunca executa docking, e
  classifica componentes apenas por sinais mecânicos já declarados pelo
  formato (água / íon metálico / resíduo polimérico modificado / outro
  não-polímero / desconhecido) — nunca por função biológica.

## O que explicitamente não está implementado

- Preparação química de qualquer tipo: atribuição de protonação, cálculo de
  pKa, cargas parciais, minimização de geometria, solvatação, membrana.
- Geração de conformação — o comando de registro de conformer aceita uma
  geometria que o cientista já possui; nada na plataforma a computa, otimiza
  ou valida.
- Docking, scoring, ranqueamento de poses, ou definição de espaço de busca.
- Materialização estrutural de uma seleção de receptor em um novo arquivo de
  coordenadas.
- Qualquer decisão científica automática ou guiada por IA (escolha de
  tautômero, de estado de protonação, de conformação "melhor", de sítio de
  ligação).

## Objetivos futuros (não implementados, não prometidos)

A linha de pesquisa planeja, em incrementos futuros ainda não especificados
em detalhe ou não implementados, abordar preparação química real,
materialização estrutural de receptores selecionados, e possivelmente
etapas de docking — cada uma delas sujeita ao mesmo processo de
especificação e revisão adversarial descrito acima antes de qualquer
implementação. Nenhuma data, nenhum compromisso de entrega e nenhuma
afirmação de viabilidade científica é feita aqui sobre essas etapas futuras.

## Para quem avalia este projeto

Esta página não afirma que qualquer pesquisador, laboratório ou disciplina
específica estuda alguma das substâncias usadas como exemplo no tutorial
público deste repositório (cafeína, ácido rosmarínico, imidacloprido,
diclofenaco, atrazina) — essas substâncias foram escolhidas apenas por
serem publicamente documentadas no PubChem e por ilustrarem casos reais de
identificação de composto (incluindo armadilhas de identidade, como CIDs
numericamente próximos referindo-se a compostos completamente diferentes).
O tutorial reproduzível completo, com cada comando derivado do código e dos
testes reais deste repositório, está em
[`docs/tutorial-pubchem-caffeine.md`](tutorial-pubchem-caffeine.md).
