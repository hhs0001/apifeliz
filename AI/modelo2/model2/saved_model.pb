??.
?0?0
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements(
handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??+
??
ConstConst*
_output_shapes	
:?*
dtype0	*?>
value?>B?>	?"?>                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?7
Const_1Const*
_output_shapes	
:?*
dtype0*?7
value?7B?7?BmodiBtheBandByouBforBthisBareBnotBindiaBwillBthatBhasBhaveBwithBallBwasBwhatBbutBbjpBnarendraBwhoBfromBwhyByourBonlyBcongressBpeopleBhisBcanBlikeBmodisBitsBoneBourBdontBwhenBaboutBnowBvoteBtheyBsirBhowBgovtBshouldBjustBdidBrahulBelectionBindianBthereBknowBafterBmoreBsaidBtimeBtBwantBevenBmBaBalsoBgoodBsaysBanyBagainBseeBnationBprimeBministerBhadBgandhiBthinkByesBveryBgiveByearsBsayBbecauseBsBtheseBniravBpleaseBgreatBgetBthenB
governmentBspaceBoutBagainstBeveryBcountryBthB	chowkidarB2019BnewBbeenBneverBmodi’BneedBunderBwBsomeBcreditBmoBfirstBdoneBdrdoBwouldBpakistanBtodayBcBwhereBbestBaddressBwhichBmakeBmanyBpowerBwellBrightBsameBhimBmodB
oppositionBwereBsupportBthanB2014BmuchBtakeBbackBcomeBdownBproudBpB’BmadeBliveBdearBtheirBhaiBmanBmayBnewsBpartyBwinBleaderBnehruBantiBgoingBmissionBbBmustB	electionsBworkBmediaBlastBthoseBdoesBstillBmoneyBfBaskBsuchBcongratulationsBdayBhBmostBbeforeBalwaysBthatsBbigBwatchBcantBtrueBreallyBsheBnB
scientistsBwayBbeingBgotBdidntBoverBtooBhereBdoingBworldBloveB	satelliteBshahBdBthingBanotherBpoorBhateBbetterBupaBnothingBnextBsayingBquestionBindiansBgivenBanBalreadyBnameBthanksB	interviewBotherBbecomeBspeechBtellBisroBiBstopBletBonceBdoesntBshaktiBcampaignBjobBgaveBwantsBcouldB
understandB	politicalBthankBindiasBdon’BrBgBfakeBpersonBgivingBshowBschemeBthemBreadBlookBsinceBduringBfamilyBrealBsureBcalledBtwoBsurgicalBbelieveBchorBlBasatBkeepBperBlokBshriBwhileBbhaktsBeverBanythingBhopeBpappuBcroreBagreeBjobsByearBtalkingBcoBpromisedBstrikeBbhiBtweetBmakingBeBvideoBlakhBsabhaBteamBpublicBeveryoneBchiefBairBachievementBpoliticsBwrongBhappyBrallyBintoBcongBbothBinBnationalBtryingBbetweenBamitBseemsBreasonBoBherBcontestBtalkBshotBleadersBguysBknowsBjaiB
everythingBcourtBpplByBfullBannouncementBneedsBnamoBstartedBfoBstateBhaBseenBfailedBbiopicBmeansBthingsBwaitingBbailBgoesBpromiseBbhaktBaskedBpakBfightBownBfeelBsarkarBdevelopmentBtestBproblemBhinduBfewBlolBdueBwholeBwaveBbiggestB	announcedBwontBgodB
differenceBwhBcallBvaranasiBthoughtBstrongBseatsB
commissionBlostB	somethingBshameBcomingBanswerBtookBcaseBtakingB	scientistBaskingBhimselfBdaysBwiBusedBlakhsBoffBfactBbhaiBarmyBgettingBactuallyBhardBrememberBanyoneBuseBwithoutBmeanB	importantBpostBthaBpriyankaBcomesBpointB100BpromisesBsinghBsuccessBputBworryBworkingBforgetBtoldBruleBhighBpartBrssBchangeBbadBoldB	statementBguyBplsBbharatBcommonBawayBliesBgivesBfaceB
corruptionBmissileBrespectBmainBentireBtakenByoBlifeBcameBarnabBlongBchinaBndaBletsBleftBhelpBmadamBhappenedBsomeoneBbringBwishBfreeBcheckBcorruptBshootsBtimesBmovieBissueBvotersBtillBpollBpartiesBfarBstartBtruthBsuccessfullyBlotBfanB	supporterBscamBreBmodijiBwarBsaBeconomyByeahBwhateverBindiBviaB	questionsBcodeBaapBproofBcleanBtaxBsetBrequestBbankBjoinBincomeBdoubtBlooksBgujaratBfarmersBdelhiBnoBfoolBmallyaBexactlyB	candidateBblameBvisionBvijayBmindBjeeBenoughBclearBheyBfilmBvotesBtakesBstoryBseatBresponsibleBindia’BattackByouthBtryBsouthBnaBmaBwowBwinsBdefeatBannounceByrsBcannotBusingBtrustBmomentBpossibleBmessageBreportBsuperBshowsBgetsBreadyBragaBwentBfiveBwhatsBtopBproBmatterBfriendsBformerBbroBtwitterBrajanBindiraB
absolutelyB	presidentB
leadershipBmakesBindBheardB“BsoonBsaveBprBmuslimsBimranBdidn’BnyayBelseBcaBsorryBplzBkindB	addressesBwaBfindBconBmodelBmightB
successfulBhistoryBbeBassamBwaitByetBvotingBleastBwonderB	respectedB
supportingBsawBsafeBhiB2012BthiBreasonsBisntBtermBscaredBlistenBwantedBrunBspeakBmeetingBblackBlowBkhanBfearBthroughBsingleBsecondBhugeBledBcreatedBvBwelcomeBmuslimBwordsBgovBwonBgoB	democracyB
supportersBhonestBwilBinsteadBfutureBstupidBshareBniceBhindBhatredBbalakotBantisatelliteBnobodyBcorrectBantimodiBhappenBbecameBideaBdeBpolicyBexpectBchannelBwomenBmeerutBkBhavingBaurBuBissuesBstandBpaBkyaBbehindBaccountBpovertyBcitizensBcareBshBharBdecisionBarticleByogiBpressBpaidBopenBlondonBarBableBunemploymentBreplyBlaunchBcongratsBagendaB	accordingBtrailerBstBlogicBdebateBsuBstrikesBfollowBwordBthat’BpoBlistBbecomesBfriendBbrotherBsecurityBleaveBhesBfinallyBjokeBfatherBeachBcreateBwhetherBvotedBcourageBactionByoungBreleaseBnoticeBmamBalBnahiBjBhatersBconductBhitBaroundBtrumpBtowardsBjailB
definitelyB	destroyedBclaimBdefenceBcan’BthinksBlookingBlevelBcroresBchB4thBwatchingBthinkingBplaceBjumlaBjisBactB‘ByoureBbusyBbroughtBadvaniB
terroristsBsimpleBplanBgrowthBchanceBtweetsBendBseBpastBearlierBprojectB
addressingB	terroristBremoveBralliesBpollsBdeshBbjpsBloanBsaluteBramBfailureBsoniaBraghuramBforeignBeraB
chowkidarsBchooseBspeakingBnorthBdoBabtB72000BmajorityBmajorBambaniBupdatesBneBkumarBblindBloseB
journalistBhearBgutsBaheadBpayBhindusBdataBactorBtellingBrafaleBjawanB
contestingB
capabilityBwriteBtotallyBfeelingBelectBbanBgameB	desperateBtohBshownBmasterBkarBdoesn’B6000BpulwamaBneitherBguessBcitizenB	seriouslyBsendBrichBpeBselfBlivingBhavBfactsBbreakingBatleastBrealityBincBidiotBhelloBhellBhandBeducatedBthoughBnumberBcourseBcallsBaprilBopinionBhappensBsoldBdynastyB	yesterdayBwoBminimumBkashmirB
propagandaBforcesB	differentBcommentBchitBsadBplayingBodishaBliBeliteBworstBstarBsmallBothersBmarchBfightingBdecidedBseriesBmonthBmiBlearnBjaitleyBhahaBgoneBdramaBcallingBrunningBmoveBclearlyBwinningBsideBjammuB	educationBdreamBdeniedBbuB1stBsocialBliberalsBlaunchedBhearingBgangBclassBcardB	announcesBvisitBvictoryBgroundBfrBcozBchoiceBraBproveBmumkinBgandhisBfaBelBeconomicBspeaksBlessBkindlyBhomeBacrossBtalksBschemesBmanmohanBkilledBgeneralBchangedBcbiBbusinessBafraidBaddBnorBmorningBleadBwitBteaB	shamelessBpoliticiansBmassiveBlandBacceptBmiddleBmeinBlieBlawBknownB	karnatakaBfirBworksBseriousBrajdeepBpresentBnoneBlatestBlateBhahahaBfunnyBeasyBadaniBtriedBtodaysBstayBrepublicB	pakistaniBnoteBmeetBheadBgovtsBexceptBbuyBाBstoppedBhainBachievementsBsellingBseeingBmaybeBjoshiB	excellentBdecideB	dangerousBclaimsBalongByourselfBworriedBvoterBevBbsfBagoBstepBpleaBofficialBfalseBasksBswamyBslamsBsenseBrestB
politicianBlanguageBfoundBeastBdestroyBtamilBsonBnitiBmistakeBhandsBcreditedBthreeBloBitselfBformBexpectedBdeathBpradeshBkillBwahBticketBstandsBsabBknewBgroupBfroB	followingBcrBbaBviewBtejasviBplayBmakersBlineBkuchBbuiltB	bangaloreBattacksBachievedBstates
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
H
Const_4Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
Adam/lstm_2/lstm_cell_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_2/lstm_cell_6/bias/v
?
2Adam/lstm_2/lstm_cell_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_6/bias/v*
_output_shapes	
:?*
dtype0
?
*Adam/lstm_2/lstm_cell_6/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*;
shared_name,*Adam/lstm_2/lstm_cell_6/recurrent_kernel/v
?
>Adam/lstm_2/lstm_cell_6/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_6/recurrent_kernel/v*
_output_shapes
:	d?*
dtype0
?
 Adam/lstm_2/lstm_cell_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*1
shared_name" Adam/lstm_2/lstm_cell_6/kernel/v
?
4Adam/lstm_2/lstm_cell_6/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_6/kernel/v*
_output_shapes
:	@?*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_3/kernel/v
?
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	?*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	d?*
dtype0
?
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*.
shared_nameAdam/embedding_1/embeddings/v
?
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes
:	?@*
dtype0
?
Adam/lstm_2/lstm_cell_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_2/lstm_cell_6/bias/m
?
2Adam/lstm_2/lstm_cell_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_6/bias/m*
_output_shapes	
:?*
dtype0
?
*Adam/lstm_2/lstm_cell_6/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*;
shared_name,*Adam/lstm_2/lstm_cell_6/recurrent_kernel/m
?
>Adam/lstm_2/lstm_cell_6/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_6/recurrent_kernel/m*
_output_shapes
:	d?*
dtype0
?
 Adam/lstm_2/lstm_cell_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*1
shared_name" Adam/lstm_2/lstm_cell_6/kernel/m
?
4Adam/lstm_2/lstm_cell_6/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_6/kernel/m*
_output_shapes
:	@?*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_3/kernel/m
?
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	?*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	d?*
dtype0
?
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*.
shared_nameAdam/embedding_1/embeddings/m
?
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes
:	?@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name8292*
value_dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
?
lstm_2/lstm_cell_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_2/lstm_cell_6/bias
?
+lstm_2/lstm_cell_6/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_6/bias*
_output_shapes	
:?*
dtype0
?
#lstm_2/lstm_cell_6/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*4
shared_name%#lstm_2/lstm_cell_6/recurrent_kernel
?
7lstm_2/lstm_cell_6/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_6/recurrent_kernel*
_output_shapes
:	d?*
dtype0
?
lstm_2/lstm_cell_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?**
shared_namelstm_2/lstm_cell_6/kernel
?
-lstm_2/lstm_cell_6/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_6/kernel*
_output_shapes
:	@?*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	d?*
dtype0
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	?@*
dtype0
?
(serving_default_text_vectorization_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall(serving_default_text_vectorization_input
hash_tableConst_5Const_4Const_3embedding_1/embeddingslstm_2/lstm_cell_6/kernellstm_2/lstm_cell_6/bias#lstm_2/lstm_cell_6/recurrent_kerneldense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_607294
?
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__initializer_609918
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__initializer_609933
:
NoOpNoOp^PartitionedCall^StatefulPartitionedCall_1
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?M
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?M
value?MB?M B?M
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
;
	keras_api
_lookup_layer
_adapt_function*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _random_generator
!cell
"
state_spec*
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_random_generator* 
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias*
<
1
:2
;3
<4
)5
*6
87
98*
<
0
:1
;2
<3
)4
*5
86
97*
* 
?
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Btrace_0
Ctrace_1
Dtrace_2
Etrace_3* 
6
Ftrace_0
Gtrace_1
Htrace_2
Itrace_3* 
/
J	capture_1
K	capture_2
L	capture_3* 
?
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_ratem?)m?*m?8m?9m?:m?;m?<m?v?)v?*v?8v?9v?:v?;v?<v?*

Rserving_default* 
* 
7
S	keras_api
Tlookup_table
Utoken_counts*

Vtrace_0* 

0*

0*
* 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

\trace_0* 

]trace_0* 
jd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

:0
;1
<2*

:0
;1
<2*
* 
?

^states
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
dtrace_0
etrace_1
ftrace_2
gtrace_3* 
6
htrace_0
itrace_1
jtrace_2
ktrace_3* 
* 
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses
r_random_generator
s
state_size

:kernel
;recurrent_kernel
<bias*
* 

)0
*1*

)0
*1*
* 
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

ytrace_0* 

ztrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

80
91*

80
91*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_2/lstm_cell_6/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_2/lstm_cell_6/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_2/lstm_cell_6/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

?0
?1*
* 
* 
/
J	capture_1
K	capture_2
L	capture_3* 
/
J	capture_1
K	capture_2
L	capture_3* 
/
J	capture_1
K	capture_2
L	capture_3* 
/
J	capture_1
K	capture_2
L	capture_3* 
/
J	capture_1
K	capture_2
L	capture_3* 
/
J	capture_1
K	capture_2
L	capture_3* 
/
J	capture_1
K	capture_2
L	capture_3* 
/
J	capture_1
K	capture_2
L	capture_3* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
/
J	capture_1
K	capture_2
L	capture_3* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
?
?_create_resource
?_initialize
?_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*

?	capture_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

!0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

:0
;1
<2*

:0
;1
<2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
"
?	capture_1
?	capture_2* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_2/lstm_cell_6/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_6/recurrent_kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_2/lstm_cell_6/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_2/lstm_cell_6/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_6/recurrent_kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_2/lstm_cell_6/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp-lstm_2/lstm_cell_6/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_6/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_6/kernel/m/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_6/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_6/bias/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_6/kernel/v/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_6/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_6/bias/v/Read/ReadVariableOpConst_6*0
Tin)
'2%		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_610102
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/biaslstm_2/lstm_cell_6/kernel#lstm_2/lstm_cell_6/recurrent_kernellstm_2/lstm_cell_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotal_1count_1totalcountAdam/embedding_1/embeddings/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/m Adam/lstm_2/lstm_cell_6/kernel/m*Adam/lstm_2/lstm_cell_6/recurrent_kernel/mAdam/lstm_2/lstm_cell_6/bias/mAdam/embedding_1/embeddings/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v Adam/lstm_2/lstm_cell_6/kernel/v*Adam/lstm_2/lstm_cell_6/recurrent_kernel/vAdam/lstm_2/lstm_cell_6/bias/v*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_610223??)
?~
?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_605907

inputs

states
states_10
split_readvariableop_resource:	@?.
split_1_readvariableop_resource:	?*
readvariableop_resource:	d?
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????@O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????dT
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dS
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????ds
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????do
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????dT
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dS
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????ds
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????do
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????dT
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dS
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????ds
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????do
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????dT
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dS
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????ds
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????do
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????dW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@[
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@[
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@[
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:?????????d_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:?????????d_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:?????????d_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:?????????dS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????dl
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????dl
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????dl
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????d[
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d[
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d[
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d[
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????dg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????di
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????dh
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????dW
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????di
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????dh
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????dU
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????dV
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:?????????di
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????dh
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????dQ
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????dK
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????dZ
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????d[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????dZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:?????????d:?????????d: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_606408

inputs1
matmul_readvariableop_resource:	d?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_609611

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
while_cond_609021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_14
0while_while_cond_609021___redundant_placeholder04
0while_while_cond_609021___redundant_placeholder14
0while_while_cond_609021___redundant_placeholder24
0while_while_cond_609021___redundant_placeholder34
0while_while_cond_609021___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :?????????d:?????????d:?????????d: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_609596

inputs1
matmul_readvariableop_resource:	d?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_609175

inputs
mask
<
)lstm_cell_6_split_readvariableop_resource:	@?:
+lstm_cell_6_split_1_readvariableop_resource:	?6
#lstm_cell_6_readvariableop_resource:	d?
identity??lstm_cell_6/ReadVariableOp?lstm_cell_6/ReadVariableOp_1?lstm_cell_6/ReadVariableOp_2?lstm_cell_6/ReadVariableOp_3? lstm_cell_6/split/ReadVariableOp?"lstm_cell_6/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????v

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskc
lstm_cell_6/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_likeFill$lstm_cell_6/ones_like/Shape:output:0$lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@[
lstm_cell_6/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_like_1Fill&lstm_cell_6/ones_like_1/Shape:output:0&lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/mulMulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_1Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_2Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_3Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_6/split/ReadVariableOpReadVariableOp)lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0(lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split
lstm_cell_6/MatMulMatMullstm_cell_6/mul:z:0lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_1MatMullstm_cell_6/mul_1:z:0lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_2MatMullstm_cell_6/mul_2:z:0lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_3MatMullstm_cell_6/mul_3:z:0lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????d_
lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_6/split_1/ReadVariableOpReadVariableOp+lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_6/split_1Split&lstm_cell_6/split_1/split_dim:output:0*lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
lstm_cell_6/BiasAddBiasAddlstm_cell_6/MatMul:product:0lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_1BiasAddlstm_cell_6/MatMul_1:product:0lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_2BiasAddlstm_cell_6/MatMul_2:product:0lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_3BiasAddlstm_cell_6/MatMul_3:product:0lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_4Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_5Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_6Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_7Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d
lstm_cell_6/ReadVariableOpReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0p
lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   r
!lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_sliceStridedSlice"lstm_cell_6/ReadVariableOp:value:0(lstm_cell_6/strided_slice/stack:output:0*lstm_cell_6/strided_slice/stack_1:output:0*lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_4MatMullstm_cell_6/mul_4:z:0"lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/addAddV2lstm_cell_6/BiasAdd:output:0lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????de
lstm_cell_6/SigmoidSigmoidlstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_1ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   t
#lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_1StridedSlice$lstm_cell_6/ReadVariableOp_1:value:0*lstm_cell_6/strided_slice_1/stack:output:0,lstm_cell_6/strided_slice_1/stack_1:output:0,lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_5MatMullstm_cell_6/mul_5:z:0$lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_1AddV2lstm_cell_6/BiasAdd_1:output:0lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????dw
lstm_cell_6/mul_8Mullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_2ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_2StridedSlice$lstm_cell_6/ReadVariableOp_2:value:0*lstm_cell_6/strided_slice_2/stack:output:0,lstm_cell_6/strided_slice_2/stack_1:output:0,lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_6MatMullstm_cell_6/mul_6:z:0$lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_2AddV2lstm_cell_6/BiasAdd_2:output:0lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????da
lstm_cell_6/TanhTanhlstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????dy
lstm_cell_6/mul_9Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????dz
lstm_cell_6/add_3AddV2lstm_cell_6/mul_8:z:0lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_3ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_3StridedSlice$lstm_cell_6/ReadVariableOp_3:value:0*lstm_cell_6/strided_slice_3/stack:output:0,lstm_cell_6/strided_slice_3/stack_1:output:0,lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_7MatMullstm_cell_6/mul_7:z:0$lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_4AddV2lstm_cell_6/BiasAdd_3:output:0lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????dc
lstm_cell_6/Tanh_1Tanhlstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d~
lstm_cell_6/mul_10Mullstm_cell_6/Sigmoid_2:y:0lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:???a

zeros_like	ZerosLikelstm_cell_6/mul_10:z:0*
T0*'
_output_shapes
:?????????dc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_6_split_readvariableop_resource+lstm_cell_6_split_1_readvariableop_resource#lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *
bodyR
while_body_609022*
condR
while_cond_609021*`
output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^lstm_cell_6/ReadVariableOp^lstm_cell_6/ReadVariableOp_1^lstm_cell_6/ReadVariableOp_2^lstm_cell_6/ReadVariableOp_3!^lstm_cell_6/split/ReadVariableOp#^lstm_cell_6/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????????????@:??????????????????: : : 28
lstm_cell_6/ReadVariableOplstm_cell_6/ReadVariableOp2<
lstm_cell_6/ReadVariableOp_1lstm_cell_6/ReadVariableOp_12<
lstm_cell_6/ReadVariableOp_2lstm_cell_6/ReadVariableOp_22<
lstm_cell_6/ReadVariableOp_3lstm_cell_6/ReadVariableOp_32D
 lstm_cell_6/split/ReadVariableOp lstm_cell_6/split/ReadVariableOp2H
"lstm_cell_6/split_1/ReadVariableOp"lstm_cell_6/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_606419

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_606037

inputs%
lstm_cell_6_605953:	@?!
lstm_cell_6_605955:	?%
lstm_cell_6_605957:	d?
identity??#lstm_cell_6/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
#lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_6_605953lstm_cell_6_605955lstm_cell_6_605957*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_605907n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_6_605953lstm_cell_6_605955lstm_cell_6_605957*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_605967*
condR
while_cond_605966*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????dt
NoOpNoOp$^lstm_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2J
#lstm_cell_6/StatefulPartitionedCall#lstm_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?u
?	
while_body_608394
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_6_split_readvariableop_resource_0:	@?B
3while_lstm_cell_6_split_1_readvariableop_resource_0:	?>
+while_lstm_cell_6_readvariableop_resource_0:	d?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_6_split_readvariableop_resource:	@?@
1while_lstm_cell_6_split_1_readvariableop_resource:	?<
)while_lstm_cell_6_readvariableop_resource:	d??? while/lstm_cell_6/ReadVariableOp?"while/lstm_cell_6/ReadVariableOp_1?"while/lstm_cell_6/ReadVariableOp_2?"while/lstm_cell_6/ReadVariableOp_3?&while/lstm_cell_6/split/ReadVariableOp?(while/lstm_cell_6/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
!while/lstm_cell_6/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_likeFill*while/lstm_cell_6/ones_like/Shape:output:0*while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@f
#while/lstm_cell_6/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_like_1Fill,while/lstm_cell_6/ones_like_1/Shape:output:0,while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_6/split/ReadVariableOpReadVariableOp1while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0.while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
while/lstm_cell_6/MatMulMatMulwhile/lstm_cell_6/mul:z:0 while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_1MatMulwhile/lstm_cell_6/mul_1:z:0 while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_2MatMulwhile/lstm_cell_6/mul_2:z:0 while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_3MatMulwhile/lstm_cell_6/mul_3:z:0 while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????de
#while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_6/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_6/split_1Split,while/lstm_cell_6/split_1/split_dim:output:00while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
while/lstm_cell_6/BiasAddBiasAdd"while/lstm_cell_6/MatMul:product:0"while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_1BiasAdd$while/lstm_cell_6/MatMul_1:product:0"while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_2BiasAdd$while/lstm_cell_6/MatMul_2:product:0"while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_3BiasAdd$while/lstm_cell_6/MatMul_3:product:0"while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_4Mulwhile_placeholder_2&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_5Mulwhile_placeholder_2&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_6Mulwhile_placeholder_2&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_7Mulwhile_placeholder_2&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/ReadVariableOpReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0v
%while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   x
'while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_6/strided_sliceStridedSlice(while/lstm_cell_6/ReadVariableOp:value:0.while/lstm_cell_6/strided_slice/stack:output:00while/lstm_cell_6/strided_slice/stack_1:output:00while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_4MatMulwhile/lstm_cell_6/mul_4:z:0(while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/addAddV2"while/lstm_cell_6/BiasAdd:output:0$while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????dq
while/lstm_cell_6/SigmoidSigmoidwhile/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_1ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   z
)while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_1StridedSlice*while/lstm_cell_6/ReadVariableOp_1:value:00while/lstm_cell_6/strided_slice_1/stack:output:02while/lstm_cell_6/strided_slice_1/stack_1:output:02while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_5MatMulwhile/lstm_cell_6/mul_5:z:0*while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_1AddV2$while/lstm_cell_6/BiasAdd_1:output:0$while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_1Sigmoidwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_8Mulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_2ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_2StridedSlice*while/lstm_cell_6/ReadVariableOp_2:value:00while/lstm_cell_6/strided_slice_2/stack:output:02while/lstm_cell_6/strided_slice_2/stack_1:output:02while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_6MatMulwhile/lstm_cell_6/mul_6:z:0*while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_2AddV2$while/lstm_cell_6/BiasAdd_2:output:0$while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????dm
while/lstm_cell_6/TanhTanhwhile/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_9Mulwhile/lstm_cell_6/Sigmoid:y:0while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_3AddV2while/lstm_cell_6/mul_8:z:0while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_3ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_3StridedSlice*while/lstm_cell_6/ReadVariableOp_3:value:00while/lstm_cell_6/strided_slice_3/stack:output:02while/lstm_cell_6/strided_slice_3/stack_1:output:02while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_7MatMulwhile/lstm_cell_6/mul_7:z:0*while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_4AddV2$while/lstm_cell_6/BiasAdd_3:output:0$while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_2Sigmoidwhile/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????do
while/lstm_cell_6/Tanh_1Tanhwhile/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_10Mulwhile/lstm_cell_6/Sigmoid_2:y:0while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_6/mul_10:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_6/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:?????????dx
while/Identity_5Identitywhile/lstm_cell_6/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp!^while/lstm_cell_6/ReadVariableOp#^while/lstm_cell_6/ReadVariableOp_1#^while/lstm_cell_6/ReadVariableOp_2#^while/lstm_cell_6/ReadVariableOp_3'^while/lstm_cell_6/split/ReadVariableOp)^while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_6_readvariableop_resource+while_lstm_cell_6_readvariableop_resource_0"h
1while_lstm_cell_6_split_1_readvariableop_resource3while_lstm_cell_6_split_1_readvariableop_resource_0"d
/while_lstm_cell_6_split_readvariableop_resource1while_lstm_cell_6_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????d:?????????d: : : : : 2D
 while/lstm_cell_6/ReadVariableOp while/lstm_cell_6/ReadVariableOp2H
"while/lstm_cell_6/ReadVariableOp_1"while/lstm_cell_6/ReadVariableOp_12H
"while/lstm_cell_6/ReadVariableOp_2"while/lstm_cell_6/ReadVariableOp_22H
"while/lstm_cell_6/ReadVariableOp_3"while/lstm_cell_6/ReadVariableOp_32P
&while/lstm_cell_6/split/ReadVariableOp&while/lstm_cell_6/split/ReadVariableOp2T
(while/lstm_cell_6/split_1/ReadVariableOp(while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
??
?

while_body_606708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0D
1while_lstm_cell_6_split_readvariableop_resource_0:	@?B
3while_lstm_cell_6_split_1_readvariableop_resource_0:	?>
+while_lstm_cell_6_readvariableop_resource_0:	d?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorB
/while_lstm_cell_6_split_readvariableop_resource:	@?@
1while_lstm_cell_6_split_1_readvariableop_resource:	?<
)while_lstm_cell_6_readvariableop_resource:	d??? while/lstm_cell_6/ReadVariableOp?"while/lstm_cell_6/ReadVariableOp_1?"while/lstm_cell_6/ReadVariableOp_2?"while/lstm_cell_6/ReadVariableOp_3?&while/lstm_cell_6/split/ReadVariableOp?(while/lstm_cell_6/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
!while/lstm_cell_6/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_likeFill*while/lstm_cell_6/ones_like/Shape:output:0*while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@d
while/lstm_cell_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout/MulMul$while/lstm_cell_6/ones_like:output:0(while/lstm_cell_6/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@s
while/lstm_cell_6/dropout/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_6/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_6/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0m
(while/lstm_cell_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_6/dropout/GreaterEqualGreaterEqual?while/lstm_cell_6/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/dropout/CastCast*while/lstm_cell_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
while/lstm_cell_6/dropout/Mul_1Mul!while/lstm_cell_6/dropout/Mul:z:0"while/lstm_cell_6/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_1/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_1/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_1/CastCast,while/lstm_cell_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_1/Mul_1Mul#while/lstm_cell_6/dropout_1/Mul:z:0$while/lstm_cell_6/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_2/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_2/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_2/CastCast,while/lstm_cell_6/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_2/Mul_1Mul#while/lstm_cell_6/dropout_2/Mul:z:0$while/lstm_cell_6/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_3/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_3/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_3/CastCast,while/lstm_cell_6/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_3/Mul_1Mul#while/lstm_cell_6/dropout_3/Mul:z:0$while/lstm_cell_6/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@f
#while/lstm_cell_6/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:h
#while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_like_1Fill,while/lstm_cell_6/ones_like_1/Shape:output:0,while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_4/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_4/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_4/CastCast,while/lstm_cell_6/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_4/Mul_1Mul#while/lstm_cell_6/dropout_4/Mul:z:0$while/lstm_cell_6/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_5/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_5/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_5/CastCast,while/lstm_cell_6/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_5/Mul_1Mul#while/lstm_cell_6/dropout_5/Mul:z:0$while/lstm_cell_6/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_6/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_6/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_6/CastCast,while/lstm_cell_6/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_6/Mul_1Mul#while/lstm_cell_6/dropout_6/Mul:z:0$while/lstm_cell_6/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_7/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_7/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_7/CastCast,while/lstm_cell_6/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_7/Mul_1Mul#while/lstm_cell_6/dropout_7/Mul:z:0$while/lstm_cell_6/dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_6/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_6/split/ReadVariableOpReadVariableOp1while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0.while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
while/lstm_cell_6/MatMulMatMulwhile/lstm_cell_6/mul:z:0 while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_1MatMulwhile/lstm_cell_6/mul_1:z:0 while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_2MatMulwhile/lstm_cell_6/mul_2:z:0 while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_3MatMulwhile/lstm_cell_6/mul_3:z:0 while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????de
#while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_6/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_6/split_1Split,while/lstm_cell_6/split_1/split_dim:output:00while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
while/lstm_cell_6/BiasAddBiasAdd"while/lstm_cell_6/MatMul:product:0"while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_1BiasAdd$while/lstm_cell_6/MatMul_1:product:0"while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_2BiasAdd$while/lstm_cell_6/MatMul_2:product:0"while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_3BiasAdd$while/lstm_cell_6/MatMul_3:product:0"while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_4Mulwhile_placeholder_3%while/lstm_cell_6/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_5Mulwhile_placeholder_3%while/lstm_cell_6/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_6Mulwhile_placeholder_3%while/lstm_cell_6/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_7Mulwhile_placeholder_3%while/lstm_cell_6/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/ReadVariableOpReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0v
%while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   x
'while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_6/strided_sliceStridedSlice(while/lstm_cell_6/ReadVariableOp:value:0.while/lstm_cell_6/strided_slice/stack:output:00while/lstm_cell_6/strided_slice/stack_1:output:00while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_4MatMulwhile/lstm_cell_6/mul_4:z:0(while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/addAddV2"while/lstm_cell_6/BiasAdd:output:0$while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????dq
while/lstm_cell_6/SigmoidSigmoidwhile/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_1ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   z
)while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_1StridedSlice*while/lstm_cell_6/ReadVariableOp_1:value:00while/lstm_cell_6/strided_slice_1/stack:output:02while/lstm_cell_6/strided_slice_1/stack_1:output:02while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_5MatMulwhile/lstm_cell_6/mul_5:z:0*while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_1AddV2$while/lstm_cell_6/BiasAdd_1:output:0$while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_1Sigmoidwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_8Mulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_2ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_2StridedSlice*while/lstm_cell_6/ReadVariableOp_2:value:00while/lstm_cell_6/strided_slice_2/stack:output:02while/lstm_cell_6/strided_slice_2/stack_1:output:02while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_6MatMulwhile/lstm_cell_6/mul_6:z:0*while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_2AddV2$while/lstm_cell_6/BiasAdd_2:output:0$while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????dm
while/lstm_cell_6/TanhTanhwhile/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_9Mulwhile/lstm_cell_6/Sigmoid:y:0while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_3AddV2while/lstm_cell_6/mul_8:z:0while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_3ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_3StridedSlice*while/lstm_cell_6/ReadVariableOp_3:value:00while/lstm_cell_6/strided_slice_3/stack:output:02while/lstm_cell_6/strided_slice_3/stack_1:output:02while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_7MatMulwhile/lstm_cell_6/mul_7:z:0*while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_4AddV2$while/lstm_cell_6/BiasAdd_3:output:0$while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_2Sigmoidwhile/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????do
while/lstm_cell_6/Tanh_1Tanhwhile/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_10Mulwhile/lstm_cell_6/Sigmoid_2:y:0while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????de
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_6/mul_10:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????dg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????g
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:??????????
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_6/mul_10:z:0while_placeholder_3*
T0*'
_output_shapes
:?????????d?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_6/add_3:z:0while_placeholder_4*
T0*'
_output_shapes
:?????????dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*'
_output_shapes
:?????????dv
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*'
_output_shapes
:?????????dv
while/Identity_6Identitywhile/SelectV2_2:output:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp!^while/lstm_cell_6/ReadVariableOp#^while/lstm_cell_6/ReadVariableOp_1#^while/lstm_cell_6/ReadVariableOp_2#^while/lstm_cell_6/ReadVariableOp_3'^while/lstm_cell_6/split/ReadVariableOp)^while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_6_readvariableop_resource+while_lstm_cell_6_readvariableop_resource_0"h
1while_lstm_cell_6_split_1_readvariableop_resource3while_lstm_cell_6_split_1_readvariableop_resource_0"d
/while_lstm_cell_6_split_readvariableop_resource1while_lstm_cell_6_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : 2D
 while/lstm_cell_6/ReadVariableOp while/lstm_cell_6/ReadVariableOp2H
"while/lstm_cell_6/ReadVariableOp_1"while/lstm_cell_6/ReadVariableOp_12H
"while/lstm_cell_6/ReadVariableOp_2"while/lstm_cell_6/ReadVariableOp_22H
"while/lstm_cell_6/ReadVariableOp_3"while/lstm_cell_6/ReadVariableOp_32P
&while/lstm_cell_6/split/ReadVariableOp&while/lstm_cell_6/split/ReadVariableOp2T
(while/lstm_cell_6/split_1/ReadVariableOp(while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
,__inference_embedding_1_layer_call_fn_608229

inputs	
unknown:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_606111|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?D
?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_609759

inputs
states_0
states_10
split_readvariableop_resource:	@?.
split_1_readvariableop_resource:	?*
readvariableop_resource:	d?
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????dX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????@Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????@Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????@Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:?????????d_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:?????????d_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:?????????d_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:?????????dS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????dl
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????dl
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????dl
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????d^
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????d^
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????d^
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????d^
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????dg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????di
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????dh
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????dW
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????di
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????dh
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????dU
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????dV
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:?????????di
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????dh
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????dQ
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????dK
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????dZ
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????d[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????dZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:?????????d:?????????d: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/1
?t
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607257
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	%
embedding_1_607233:	?@ 
lstm_2_607238:	@?
lstm_2_607240:	? 
lstm_2_607242:	d?!
dense_2_607245:	d?
dense_2_607247:	?!
dense_3_607251:	?
dense_3_607253:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
gtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ptext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountjtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????????????
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1_607233*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_606111X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
embedding_1/NotEqualNotEqual?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*0
_output_shapes
:???????????????????
lstm_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0lstm_2_607238lstm_2_607240lstm_2_607242*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_606925?
dense_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_2_607245dense_2_607247*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_606408?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_606496?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_607251dense_3_607253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_606432w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_1_layer_call_fn_607105
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?@
	unknown_4:	@?
	unknown_5:	?
	unknown_6:	d?
	unknown_7:	d?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_607049o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?

while_body_606236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0D
1while_lstm_cell_6_split_readvariableop_resource_0:	@?B
3while_lstm_cell_6_split_1_readvariableop_resource_0:	?>
+while_lstm_cell_6_readvariableop_resource_0:	d?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorB
/while_lstm_cell_6_split_readvariableop_resource:	@?@
1while_lstm_cell_6_split_1_readvariableop_resource:	?<
)while_lstm_cell_6_readvariableop_resource:	d??? while/lstm_cell_6/ReadVariableOp?"while/lstm_cell_6/ReadVariableOp_1?"while/lstm_cell_6/ReadVariableOp_2?"while/lstm_cell_6/ReadVariableOp_3?&while/lstm_cell_6/split/ReadVariableOp?(while/lstm_cell_6/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
!while/lstm_cell_6/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_likeFill*while/lstm_cell_6/ones_like/Shape:output:0*while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@f
#while/lstm_cell_6/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:h
#while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_like_1Fill,while/lstm_cell_6/ones_like_1/Shape:output:0,while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_6/split/ReadVariableOpReadVariableOp1while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0.while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
while/lstm_cell_6/MatMulMatMulwhile/lstm_cell_6/mul:z:0 while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_1MatMulwhile/lstm_cell_6/mul_1:z:0 while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_2MatMulwhile/lstm_cell_6/mul_2:z:0 while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_3MatMulwhile/lstm_cell_6/mul_3:z:0 while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????de
#while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_6/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_6/split_1Split,while/lstm_cell_6/split_1/split_dim:output:00while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
while/lstm_cell_6/BiasAddBiasAdd"while/lstm_cell_6/MatMul:product:0"while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_1BiasAdd$while/lstm_cell_6/MatMul_1:product:0"while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_2BiasAdd$while/lstm_cell_6/MatMul_2:product:0"while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_3BiasAdd$while/lstm_cell_6/MatMul_3:product:0"while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_4Mulwhile_placeholder_3&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_5Mulwhile_placeholder_3&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_6Mulwhile_placeholder_3&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_7Mulwhile_placeholder_3&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/ReadVariableOpReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0v
%while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   x
'while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_6/strided_sliceStridedSlice(while/lstm_cell_6/ReadVariableOp:value:0.while/lstm_cell_6/strided_slice/stack:output:00while/lstm_cell_6/strided_slice/stack_1:output:00while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_4MatMulwhile/lstm_cell_6/mul_4:z:0(while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/addAddV2"while/lstm_cell_6/BiasAdd:output:0$while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????dq
while/lstm_cell_6/SigmoidSigmoidwhile/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_1ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   z
)while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_1StridedSlice*while/lstm_cell_6/ReadVariableOp_1:value:00while/lstm_cell_6/strided_slice_1/stack:output:02while/lstm_cell_6/strided_slice_1/stack_1:output:02while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_5MatMulwhile/lstm_cell_6/mul_5:z:0*while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_1AddV2$while/lstm_cell_6/BiasAdd_1:output:0$while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_1Sigmoidwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_8Mulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_2ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_2StridedSlice*while/lstm_cell_6/ReadVariableOp_2:value:00while/lstm_cell_6/strided_slice_2/stack:output:02while/lstm_cell_6/strided_slice_2/stack_1:output:02while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_6MatMulwhile/lstm_cell_6/mul_6:z:0*while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_2AddV2$while/lstm_cell_6/BiasAdd_2:output:0$while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????dm
while/lstm_cell_6/TanhTanhwhile/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_9Mulwhile/lstm_cell_6/Sigmoid:y:0while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_3AddV2while/lstm_cell_6/mul_8:z:0while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_3ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_3StridedSlice*while/lstm_cell_6/ReadVariableOp_3:value:00while/lstm_cell_6/strided_slice_3/stack:output:02while/lstm_cell_6/strided_slice_3/stack_1:output:02while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_7MatMulwhile/lstm_cell_6/mul_7:z:0*while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_4AddV2$while/lstm_cell_6/BiasAdd_3:output:0$while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_2Sigmoidwhile/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????do
while/lstm_cell_6/Tanh_1Tanhwhile/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_10Mulwhile/lstm_cell_6/Sigmoid_2:y:0while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????de
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_6/mul_10:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????dg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????g
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:??????????
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_6/mul_10:z:0while_placeholder_3*
T0*'
_output_shapes
:?????????d?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_6/add_3:z:0while_placeholder_4*
T0*'
_output_shapes
:?????????dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*'
_output_shapes
:?????????dv
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*'
_output_shapes
:?????????dv
while/Identity_6Identitywhile/SelectV2_2:output:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp!^while/lstm_cell_6/ReadVariableOp#^while/lstm_cell_6/ReadVariableOp_1#^while/lstm_cell_6/ReadVariableOp_2#^while/lstm_cell_6/ReadVariableOp_3'^while/lstm_cell_6/split/ReadVariableOp)^while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_6_readvariableop_resource+while_lstm_cell_6_readvariableop_resource_0"h
1while_lstm_cell_6_split_1_readvariableop_resource3while_lstm_cell_6_split_1_readvariableop_resource_0"d
/while_lstm_cell_6_split_readvariableop_resource1while_lstm_cell_6_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : 2D
 while/lstm_cell_6/ReadVariableOp while/lstm_cell_6/ReadVariableOp2H
"while/lstm_cell_6/ReadVariableOp_1"while/lstm_cell_6/ReadVariableOp_12H
"while/lstm_cell_6/ReadVariableOp_2"while/lstm_cell_6/ReadVariableOp_22H
"while/lstm_cell_6/ReadVariableOp_3"while/lstm_cell_6/ReadVariableOp_32P
&while/lstm_cell_6/split/ReadVariableOp&while/lstm_cell_6/split/ReadVariableOp2T
(while/lstm_cell_6/split_1/ReadVariableOp(while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?	
?
G__inference_embedding_1_layer_call_and_return_conditional_losses_606111

inputs	*
embedding_lookup_606105:	?@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_606105inputs*
Tindices0	**
_class 
loc:@embedding_lookup/606105*4
_output_shapes"
 :??????????????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/606105*4
_output_shapes"
 :??????????????????@?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_605528
text_vectorization_input\
Xsequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle]
Ysequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	9
5sequential_1_text_vectorization_string_lookup_equal_y<
8sequential_1_text_vectorization_string_lookup_selectv2_t	C
0sequential_1_embedding_1_embedding_lookup_605237:	?@P
=sequential_1_lstm_2_lstm_cell_6_split_readvariableop_resource:	@?N
?sequential_1_lstm_2_lstm_cell_6_split_1_readvariableop_resource:	?J
7sequential_1_lstm_2_lstm_cell_6_readvariableop_resource:	d?F
3sequential_1_dense_2_matmul_readvariableop_resource:	d?C
4sequential_1_dense_2_biasadd_readvariableop_resource:	?F
3sequential_1_dense_3_matmul_readvariableop_resource:	?B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity??+sequential_1/dense_2/BiasAdd/ReadVariableOp?*sequential_1/dense_2/MatMul/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?*sequential_1/dense_3/MatMul/ReadVariableOp?)sequential_1/embedding_1/embedding_lookup?.sequential_1/lstm_2/lstm_cell_6/ReadVariableOp?0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_1?0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_2?0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_3?4sequential_1/lstm_2/lstm_cell_6/split/ReadVariableOp?6sequential_1/lstm_2/lstm_cell_6/split_1/ReadVariableOp?sequential_1/lstm_2/while?Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2y
+sequential_1/text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
2sequential_1/text_vectorization/StaticRegexReplaceStaticRegexReplace4sequential_1/text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite r
1sequential_1/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
9sequential_1/text_vectorization/StringSplit/StringSplitV2StringSplitV2;sequential_1/text_vectorization/StaticRegexReplace:output:0:sequential_1/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
?sequential_1/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Asequential_1/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Asequential_1/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
9sequential_1/text_vectorization/StringSplit/strided_sliceStridedSliceCsequential_1/text_vectorization/StringSplit/StringSplitV2:indices:0Hsequential_1/text_vectorization/StringSplit/strided_slice/stack:output:0Jsequential_1/text_vectorization/StringSplit/strided_slice/stack_1:output:0Jsequential_1/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Asequential_1/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csequential_1/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csequential_1/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential_1/text_vectorization/StringSplit/strided_slice_1StridedSliceAsequential_1/text_vectorization/StringSplit/StringSplitV2:shape:0Jsequential_1/text_vectorization/StringSplit/strided_slice_1/stack:output:0Lsequential_1/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Lsequential_1/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
bsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastBsequential_1/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastDsequential_1/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapefsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
ksequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdusequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0usequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
psequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatertsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ysequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
ksequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastrsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxfsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0wsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ssequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0usequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulosequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumhsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumhsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
tsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapefsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0}sequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
osequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountwsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0wsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
isequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumvsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
msequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
isequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2vsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Xsequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleBsequential_1/text_vectorization/StringSplit/StringSplitV2:values:0Ysequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
3sequential_1/text_vectorization/string_lookup/EqualEqualBsequential_1/text_vectorization/StringSplit/StringSplitV2:values:05sequential_1_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
6sequential_1/text_vectorization/string_lookup/SelectV2SelectV27sequential_1/text_vectorization/string_lookup/Equal:z:08sequential_1_text_vectorization_string_lookup_selectv2_tTsequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
6sequential_1/text_vectorization/string_lookup/IdentityIdentity?sequential_1/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????~
<sequential_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
4sequential_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????????????
Csequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor=sequential_1/text_vectorization/RaggedToTensor/Const:output:0?sequential_1/text_vectorization/string_lookup/Identity:output:0Esequential_1/text_vectorization/RaggedToTensor/default_value:output:0Dsequential_1/text_vectorization/StringSplit/strided_slice_1:output:0Bsequential_1/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
)sequential_1/embedding_1/embedding_lookupResourceGather0sequential_1_embedding_1_embedding_lookup_605237Lsequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/605237*4
_output_shapes"
 :??????????????????@*
dtype0?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*C
_class9
75loc:@sequential_1/embedding_1/embedding_lookup/605237*4
_output_shapes"
 :??????????????????@?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@e
#sequential_1/embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
!sequential_1/embedding_1/NotEqualNotEqualLsequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0,sequential_1/embedding_1/NotEqual/y:output:0*
T0	*0
_output_shapes
:???????????????????
sequential_1/lstm_2/ShapeShape=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:q
'sequential_1/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_1/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_1/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!sequential_1/lstm_2/strided_sliceStridedSlice"sequential_1/lstm_2/Shape:output:00sequential_1/lstm_2/strided_slice/stack:output:02sequential_1/lstm_2/strided_slice/stack_1:output:02sequential_1/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_1/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
 sequential_1/lstm_2/zeros/packedPack*sequential_1/lstm_2/strided_slice:output:0+sequential_1/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_1/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/lstm_2/zerosFill)sequential_1/lstm_2/zeros/packed:output:0(sequential_1/lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:?????????df
$sequential_1/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
"sequential_1/lstm_2/zeros_1/packedPack*sequential_1/lstm_2/strided_slice:output:0-sequential_1/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_1/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/lstm_2/zeros_1Fill+sequential_1/lstm_2/zeros_1/packed:output:0*sequential_1/lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dw
"sequential_1/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/lstm_2/transpose	Transpose=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0+sequential_1/lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@l
sequential_1/lstm_2/Shape_1Shape!sequential_1/lstm_2/transpose:y:0*
T0*
_output_shapes
:s
)sequential_1/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_2/strided_slice_1StridedSlice$sequential_1/lstm_2/Shape_1:output:02sequential_1/lstm_2/strided_slice_1/stack:output:04sequential_1/lstm_2/strided_slice_1/stack_1:output:04sequential_1/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"sequential_1/lstm_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential_1/lstm_2/ExpandDims
ExpandDims%sequential_1/embedding_1/NotEqual:z:0+sequential_1/lstm_2/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????y
$sequential_1/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/lstm_2/transpose_1	Transpose'sequential_1/lstm_2/ExpandDims:output:0-sequential_1/lstm_2/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????z
/sequential_1/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!sequential_1/lstm_2/TensorArrayV2TensorListReserve8sequential_1/lstm_2/TensorArrayV2/element_shape:output:0,sequential_1/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Isequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
;sequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_2/transpose:y:0Rsequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???s
)sequential_1/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_2/strided_slice_2StridedSlice!sequential_1/lstm_2/transpose:y:02sequential_1/lstm_2/strided_slice_2/stack:output:04sequential_1/lstm_2/strided_slice_2/stack_1:output:04sequential_1/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
/sequential_1/lstm_2/lstm_cell_6/ones_like/ShapeShape,sequential_1/lstm_2/strided_slice_2:output:0*
T0*
_output_shapes
:t
/sequential_1/lstm_2/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)sequential_1/lstm_2/lstm_cell_6/ones_likeFill8sequential_1/lstm_2/lstm_cell_6/ones_like/Shape:output:08sequential_1/lstm_2/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@?
1sequential_1/lstm_2/lstm_cell_6/ones_like_1/ShapeShape"sequential_1/lstm_2/zeros:output:0*
T0*
_output_shapes
:v
1sequential_1/lstm_2/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
+sequential_1/lstm_2/lstm_cell_6/ones_like_1Fill:sequential_1/lstm_2/lstm_cell_6/ones_like_1/Shape:output:0:sequential_1/lstm_2/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
#sequential_1/lstm_2/lstm_cell_6/mulMul,sequential_1/lstm_2/strided_slice_2:output:02sequential_1/lstm_2/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
%sequential_1/lstm_2/lstm_cell_6/mul_1Mul,sequential_1/lstm_2/strided_slice_2:output:02sequential_1/lstm_2/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
%sequential_1/lstm_2/lstm_cell_6/mul_2Mul,sequential_1/lstm_2/strided_slice_2:output:02sequential_1/lstm_2/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
%sequential_1/lstm_2/lstm_cell_6/mul_3Mul,sequential_1/lstm_2/strided_slice_2:output:02sequential_1/lstm_2/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@q
/sequential_1/lstm_2/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
4sequential_1/lstm_2/lstm_cell_6/split/ReadVariableOpReadVariableOp=sequential_1_lstm_2_lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
%sequential_1/lstm_2/lstm_cell_6/splitSplit8sequential_1/lstm_2/lstm_cell_6/split/split_dim:output:0<sequential_1/lstm_2/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
&sequential_1/lstm_2/lstm_cell_6/MatMulMatMul'sequential_1/lstm_2/lstm_cell_6/mul:z:0.sequential_1/lstm_2/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
(sequential_1/lstm_2/lstm_cell_6/MatMul_1MatMul)sequential_1/lstm_2/lstm_cell_6/mul_1:z:0.sequential_1/lstm_2/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
(sequential_1/lstm_2/lstm_cell_6/MatMul_2MatMul)sequential_1/lstm_2/lstm_cell_6/mul_2:z:0.sequential_1/lstm_2/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
(sequential_1/lstm_2/lstm_cell_6/MatMul_3MatMul)sequential_1/lstm_2/lstm_cell_6/mul_3:z:0.sequential_1/lstm_2/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????ds
1sequential_1/lstm_2/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
6sequential_1/lstm_2/lstm_cell_6/split_1/ReadVariableOpReadVariableOp?sequential_1_lstm_2_lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'sequential_1/lstm_2/lstm_cell_6/split_1Split:sequential_1/lstm_2/lstm_cell_6/split_1/split_dim:output:0>sequential_1/lstm_2/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
'sequential_1/lstm_2/lstm_cell_6/BiasAddBiasAdd0sequential_1/lstm_2/lstm_cell_6/MatMul:product:00sequential_1/lstm_2/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
)sequential_1/lstm_2/lstm_cell_6/BiasAdd_1BiasAdd2sequential_1/lstm_2/lstm_cell_6/MatMul_1:product:00sequential_1/lstm_2/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
)sequential_1/lstm_2/lstm_cell_6/BiasAdd_2BiasAdd2sequential_1/lstm_2/lstm_cell_6/MatMul_2:product:00sequential_1/lstm_2/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
)sequential_1/lstm_2/lstm_cell_6/BiasAdd_3BiasAdd2sequential_1/lstm_2/lstm_cell_6/MatMul_3:product:00sequential_1/lstm_2/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/mul_4Mul"sequential_1/lstm_2/zeros:output:04sequential_1/lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/mul_5Mul"sequential_1/lstm_2/zeros:output:04sequential_1/lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/mul_6Mul"sequential_1/lstm_2/zeros:output:04sequential_1/lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/mul_7Mul"sequential_1/lstm_2/zeros:output:04sequential_1/lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
.sequential_1/lstm_2/lstm_cell_6/ReadVariableOpReadVariableOp7sequential_1_lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
3sequential_1/lstm_2/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
5sequential_1/lstm_2/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   ?
5sequential_1/lstm_2/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
-sequential_1/lstm_2/lstm_cell_6/strided_sliceStridedSlice6sequential_1/lstm_2/lstm_cell_6/ReadVariableOp:value:0<sequential_1/lstm_2/lstm_cell_6/strided_slice/stack:output:0>sequential_1/lstm_2/lstm_cell_6/strided_slice/stack_1:output:0>sequential_1/lstm_2/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
(sequential_1/lstm_2/lstm_cell_6/MatMul_4MatMul)sequential_1/lstm_2/lstm_cell_6/mul_4:z:06sequential_1/lstm_2/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
#sequential_1/lstm_2/lstm_cell_6/addAddV20sequential_1/lstm_2/lstm_cell_6/BiasAdd:output:02sequential_1/lstm_2/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????d?
'sequential_1/lstm_2/lstm_cell_6/SigmoidSigmoid'sequential_1/lstm_2/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_1ReadVariableOp7sequential_1_lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
5sequential_1/lstm_2/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   ?
7sequential_1/lstm_2/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   ?
7sequential_1/lstm_2/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_2/lstm_cell_6/strided_slice_1StridedSlice8sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_1:value:0>sequential_1/lstm_2/lstm_cell_6/strided_slice_1/stack:output:0@sequential_1/lstm_2/lstm_cell_6/strided_slice_1/stack_1:output:0@sequential_1/lstm_2/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
(sequential_1/lstm_2/lstm_cell_6/MatMul_5MatMul)sequential_1/lstm_2/lstm_cell_6/mul_5:z:08sequential_1/lstm_2/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/add_1AddV22sequential_1/lstm_2/lstm_cell_6/BiasAdd_1:output:02sequential_1/lstm_2/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????d?
)sequential_1/lstm_2/lstm_cell_6/Sigmoid_1Sigmoid)sequential_1/lstm_2/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/mul_8Mul-sequential_1/lstm_2/lstm_cell_6/Sigmoid_1:y:0$sequential_1/lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_2ReadVariableOp7sequential_1_lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
5sequential_1/lstm_2/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
7sequential_1/lstm_2/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  ?
7sequential_1/lstm_2/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_2/lstm_cell_6/strided_slice_2StridedSlice8sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_2:value:0>sequential_1/lstm_2/lstm_cell_6/strided_slice_2/stack:output:0@sequential_1/lstm_2/lstm_cell_6/strided_slice_2/stack_1:output:0@sequential_1/lstm_2/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
(sequential_1/lstm_2/lstm_cell_6/MatMul_6MatMul)sequential_1/lstm_2/lstm_cell_6/mul_6:z:08sequential_1/lstm_2/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/add_2AddV22sequential_1/lstm_2/lstm_cell_6/BiasAdd_2:output:02sequential_1/lstm_2/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????d?
$sequential_1/lstm_2/lstm_cell_6/TanhTanh)sequential_1/lstm_2/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/mul_9Mul+sequential_1/lstm_2/lstm_cell_6/Sigmoid:y:0(sequential_1/lstm_2/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/add_3AddV2)sequential_1/lstm_2/lstm_cell_6/mul_8:z:0)sequential_1/lstm_2/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_3ReadVariableOp7sequential_1_lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
5sequential_1/lstm_2/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  ?
7sequential_1/lstm_2/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
7sequential_1/lstm_2/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/sequential_1/lstm_2/lstm_cell_6/strided_slice_3StridedSlice8sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_3:value:0>sequential_1/lstm_2/lstm_cell_6/strided_slice_3/stack:output:0@sequential_1/lstm_2/lstm_cell_6/strided_slice_3/stack_1:output:0@sequential_1/lstm_2/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
(sequential_1/lstm_2/lstm_cell_6/MatMul_7MatMul)sequential_1/lstm_2/lstm_cell_6/mul_7:z:08sequential_1/lstm_2/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
%sequential_1/lstm_2/lstm_cell_6/add_4AddV22sequential_1/lstm_2/lstm_cell_6/BiasAdd_3:output:02sequential_1/lstm_2/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????d?
)sequential_1/lstm_2/lstm_cell_6/Sigmoid_2Sigmoid)sequential_1/lstm_2/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????d?
&sequential_1/lstm_2/lstm_cell_6/Tanh_1Tanh)sequential_1/lstm_2/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
&sequential_1/lstm_2/lstm_cell_6/mul_10Mul-sequential_1/lstm_2/lstm_cell_6/Sigmoid_2:y:0*sequential_1/lstm_2/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????d?
1sequential_1/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   r
0sequential_1/lstm_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_1/lstm_2/TensorArrayV2_1TensorListReserve:sequential_1/lstm_2/TensorArrayV2_1/element_shape:output:09sequential_1/lstm_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???Z
sequential_1/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : |
1sequential_1/lstm_2/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#sequential_1/lstm_2/TensorArrayV2_2TensorListReserve:sequential_1/lstm_2/TensorArrayV2_2/element_shape:output:0,sequential_1/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
Ksequential_1/lstm_2/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
=sequential_1/lstm_2/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor#sequential_1/lstm_2/transpose_1:y:0Tsequential_1/lstm_2/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
sequential_1/lstm_2/zeros_like	ZerosLike*sequential_1/lstm_2/lstm_cell_6/mul_10:z:0*
T0*'
_output_shapes
:?????????dw
,sequential_1/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????h
&sequential_1/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?	
sequential_1/lstm_2/whileWhile/sequential_1/lstm_2/while/loop_counter:output:05sequential_1/lstm_2/while/maximum_iterations:output:0!sequential_1/lstm_2/time:output:0,sequential_1/lstm_2/TensorArrayV2_1:handle:0"sequential_1/lstm_2/zeros_like:y:0"sequential_1/lstm_2/zeros:output:0$sequential_1/lstm_2/zeros_1:output:0,sequential_1/lstm_2/strided_slice_1:output:0Ksequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_1/lstm_2/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0=sequential_1_lstm_2_lstm_cell_6_split_readvariableop_resource?sequential_1_lstm_2_lstm_cell_6_split_1_readvariableop_resource7sequential_1_lstm_2_lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *1
body)R'
%sequential_1_lstm_2_while_body_605360*1
cond)R'
%sequential_1_lstm_2_while_cond_605359*`
output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *
parallel_iterations ?
Dsequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
6sequential_1/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_2/while:output:3Msequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elements|
)sequential_1/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????u
+sequential_1/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#sequential_1/lstm_2/strided_slice_3StridedSlice?sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_2/strided_slice_3/stack:output:04sequential_1/lstm_2/strided_slice_3/stack_1:output:04sequential_1/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_masky
$sequential_1/lstm_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
sequential_1/lstm_2/transpose_2	Transpose?sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????do
sequential_1/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
sequential_1/dense_2/MatMulMatMul,sequential_1/lstm_2/strided_slice_3:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
sequential_1/dropout_1/IdentityIdentity'sequential_1/dense_2/Relu:activations:0*
T0*(
_output_shapes
:???????????
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*^sequential_1/embedding_1/embedding_lookup/^sequential_1/lstm_2/lstm_cell_6/ReadVariableOp1^sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_11^sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_21^sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_35^sequential_1/lstm_2/lstm_cell_6/split/ReadVariableOp7^sequential_1/lstm_2/lstm_cell_6/split_1/ReadVariableOp^sequential_1/lstm_2/whileL^sequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2`
.sequential_1/lstm_2/lstm_cell_6/ReadVariableOp.sequential_1/lstm_2/lstm_cell_6/ReadVariableOp2d
0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_10sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_12d
0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_20sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_22d
0sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_30sequential_1/lstm_2/lstm_cell_6/ReadVariableOp_32l
4sequential_1/lstm_2/lstm_cell_6/split/ReadVariableOp4sequential_1/lstm_2/lstm_cell_6/split/ReadVariableOp2p
6sequential_1/lstm_2/lstm_cell_6/split_1/ReadVariableOp6sequential_1/lstm_2/lstm_cell_6/split_1/ReadVariableOp26
sequential_1/lstm_2/whilesequential_1/lstm_2/while2?
Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?

H__inference_sequential_1_layer_call_and_return_conditional_losses_608222

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	6
#embedding_1_embedding_lookup_607796:	?@C
0lstm_2_lstm_cell_6_split_readvariableop_resource:	@?A
2lstm_2_lstm_cell_6_split_1_readvariableop_resource:	?=
*lstm_2_lstm_cell_6_readvariableop_resource:	d?9
&dense_2_matmul_readvariableop_resource:	d?6
'dense_2_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookup?!lstm_2/lstm_cell_6/ReadVariableOp?#lstm_2/lstm_cell_6/ReadVariableOp_1?#lstm_2/lstm_cell_6/ReadVariableOp_2?#lstm_2/lstm_cell_6/ReadVariableOp_3?'lstm_2/lstm_cell_6/split/ReadVariableOp?)lstm_2/lstm_cell_6/split_1/ReadVariableOp?lstm_2/while?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
gtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ptext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountjtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????????????
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_607796?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*6
_class,
*(loc:@embedding_1/embedding_lookup/607796*4
_output_shapes"
 :??????????????????@*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/607796*4
_output_shapes"
 :??????????????????@?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
embedding_1/NotEqualNotEqual?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*0
_output_shapes
:??????????????????l
lstm_2/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dY
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dj
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@R
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
lstm_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/ExpandDims
ExpandDimsembedding_1/NotEqual:z:0lstm_2/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????l
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose_1	Transposelstm_2/ExpandDims:output:0 lstm_2/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????m
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskq
"lstm_2/lstm_cell_6/ones_like/ShapeShapelstm_2/strided_slice_2:output:0*
T0*
_output_shapes
:g
"lstm_2/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_2/lstm_cell_6/ones_likeFill+lstm_2/lstm_cell_6/ones_like/Shape:output:0+lstm_2/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@e
 lstm_2/lstm_cell_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_2/lstm_cell_6/dropout/MulMul%lstm_2/lstm_cell_6/ones_like:output:0)lstm_2/lstm_cell_6/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@u
 lstm_2/lstm_cell_6/dropout/ShapeShape%lstm_2/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
7lstm_2/lstm_cell_6/dropout/random_uniform/RandomUniformRandomUniform)lstm_2/lstm_cell_6/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0n
)lstm_2/lstm_cell_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
'lstm_2/lstm_cell_6/dropout/GreaterEqualGreaterEqual@lstm_2/lstm_cell_6/dropout/random_uniform/RandomUniform:output:02lstm_2/lstm_cell_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_2/lstm_cell_6/dropout/CastCast+lstm_2/lstm_cell_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
 lstm_2/lstm_cell_6/dropout/Mul_1Mul"lstm_2/lstm_cell_6/dropout/Mul:z:0#lstm_2/lstm_cell_6/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@g
"lstm_2/lstm_cell_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_2/lstm_cell_6/dropout_1/MulMul%lstm_2/lstm_cell_6/ones_like:output:0+lstm_2/lstm_cell_6/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@w
"lstm_2/lstm_cell_6/dropout_1/ShapeShape%lstm_2/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_2/lstm_cell_6/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_6/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0p
+lstm_2/lstm_cell_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_2/lstm_cell_6/dropout_1/GreaterEqualGreaterEqualBlstm_2/lstm_cell_6/dropout_1/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_6/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
!lstm_2/lstm_cell_6/dropout_1/CastCast-lstm_2/lstm_cell_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
"lstm_2/lstm_cell_6/dropout_1/Mul_1Mul$lstm_2/lstm_cell_6/dropout_1/Mul:z:0%lstm_2/lstm_cell_6/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@g
"lstm_2/lstm_cell_6/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_2/lstm_cell_6/dropout_2/MulMul%lstm_2/lstm_cell_6/ones_like:output:0+lstm_2/lstm_cell_6/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@w
"lstm_2/lstm_cell_6/dropout_2/ShapeShape%lstm_2/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_2/lstm_cell_6/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_6/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0p
+lstm_2/lstm_cell_6/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_2/lstm_cell_6/dropout_2/GreaterEqualGreaterEqualBlstm_2/lstm_cell_6/dropout_2/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_6/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
!lstm_2/lstm_cell_6/dropout_2/CastCast-lstm_2/lstm_cell_6/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
"lstm_2/lstm_cell_6/dropout_2/Mul_1Mul$lstm_2/lstm_cell_6/dropout_2/Mul:z:0%lstm_2/lstm_cell_6/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@g
"lstm_2/lstm_cell_6/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_2/lstm_cell_6/dropout_3/MulMul%lstm_2/lstm_cell_6/ones_like:output:0+lstm_2/lstm_cell_6/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@w
"lstm_2/lstm_cell_6/dropout_3/ShapeShape%lstm_2/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
9lstm_2/lstm_cell_6/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_6/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0p
+lstm_2/lstm_cell_6/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_2/lstm_cell_6/dropout_3/GreaterEqualGreaterEqualBlstm_2/lstm_cell_6/dropout_3/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_6/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
!lstm_2/lstm_cell_6/dropout_3/CastCast-lstm_2/lstm_cell_6/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
"lstm_2/lstm_cell_6/dropout_3/Mul_1Mul$lstm_2/lstm_cell_6/dropout_3/Mul:z:0%lstm_2/lstm_cell_6/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@i
$lstm_2/lstm_cell_6/ones_like_1/ShapeShapelstm_2/zeros:output:0*
T0*
_output_shapes
:i
$lstm_2/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_2/lstm_cell_6/ones_like_1Fill-lstm_2/lstm_cell_6/ones_like_1/Shape:output:0-lstm_2/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????dg
"lstm_2/lstm_cell_6/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_2/lstm_cell_6/dropout_4/MulMul'lstm_2/lstm_cell_6/ones_like_1:output:0+lstm_2/lstm_cell_6/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dy
"lstm_2/lstm_cell_6/dropout_4/ShapeShape'lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
9lstm_2/lstm_cell_6/dropout_4/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_6/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0p
+lstm_2/lstm_cell_6/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_2/lstm_cell_6/dropout_4/GreaterEqualGreaterEqualBlstm_2/lstm_cell_6/dropout_4/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_6/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
!lstm_2/lstm_cell_6/dropout_4/CastCast-lstm_2/lstm_cell_6/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
"lstm_2/lstm_cell_6/dropout_4/Mul_1Mul$lstm_2/lstm_cell_6/dropout_4/Mul:z:0%lstm_2/lstm_cell_6/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????dg
"lstm_2/lstm_cell_6/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_2/lstm_cell_6/dropout_5/MulMul'lstm_2/lstm_cell_6/ones_like_1:output:0+lstm_2/lstm_cell_6/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dy
"lstm_2/lstm_cell_6/dropout_5/ShapeShape'lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
9lstm_2/lstm_cell_6/dropout_5/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_6/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0p
+lstm_2/lstm_cell_6/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_2/lstm_cell_6/dropout_5/GreaterEqualGreaterEqualBlstm_2/lstm_cell_6/dropout_5/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_6/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
!lstm_2/lstm_cell_6/dropout_5/CastCast-lstm_2/lstm_cell_6/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
"lstm_2/lstm_cell_6/dropout_5/Mul_1Mul$lstm_2/lstm_cell_6/dropout_5/Mul:z:0%lstm_2/lstm_cell_6/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????dg
"lstm_2/lstm_cell_6/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_2/lstm_cell_6/dropout_6/MulMul'lstm_2/lstm_cell_6/ones_like_1:output:0+lstm_2/lstm_cell_6/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dy
"lstm_2/lstm_cell_6/dropout_6/ShapeShape'lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
9lstm_2/lstm_cell_6/dropout_6/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_6/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0p
+lstm_2/lstm_cell_6/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_2/lstm_cell_6/dropout_6/GreaterEqualGreaterEqualBlstm_2/lstm_cell_6/dropout_6/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_6/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
!lstm_2/lstm_cell_6/dropout_6/CastCast-lstm_2/lstm_cell_6/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
"lstm_2/lstm_cell_6/dropout_6/Mul_1Mul$lstm_2/lstm_cell_6/dropout_6/Mul:z:0%lstm_2/lstm_cell_6/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????dg
"lstm_2/lstm_cell_6/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 lstm_2/lstm_cell_6/dropout_7/MulMul'lstm_2/lstm_cell_6/ones_like_1:output:0+lstm_2/lstm_cell_6/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dy
"lstm_2/lstm_cell_6/dropout_7/ShapeShape'lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
9lstm_2/lstm_cell_6/dropout_7/random_uniform/RandomUniformRandomUniform+lstm_2/lstm_cell_6/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0p
+lstm_2/lstm_cell_6/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
)lstm_2/lstm_cell_6/dropout_7/GreaterEqualGreaterEqualBlstm_2/lstm_cell_6/dropout_7/random_uniform/RandomUniform:output:04lstm_2/lstm_cell_6/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
!lstm_2/lstm_cell_6/dropout_7/CastCast-lstm_2/lstm_cell_6/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
"lstm_2/lstm_cell_6/dropout_7/Mul_1Mul$lstm_2/lstm_cell_6/dropout_7/Mul:z:0%lstm_2/lstm_cell_6/dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mulMullstm_2/strided_slice_2:output:0$lstm_2/lstm_cell_6/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_2/lstm_cell_6/mul_1Mullstm_2/strided_slice_2:output:0&lstm_2/lstm_cell_6/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_2/lstm_cell_6/mul_2Mullstm_2/strided_slice_2:output:0&lstm_2/lstm_cell_6/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_2/lstm_cell_6/mul_3Mullstm_2/strided_slice_2:output:0&lstm_2/lstm_cell_6/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@d
"lstm_2/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_2/lstm_cell_6/split/ReadVariableOpReadVariableOp0lstm_2_lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_2/lstm_cell_6/splitSplit+lstm_2/lstm_cell_6/split/split_dim:output:0/lstm_2/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
lstm_2/lstm_cell_6/MatMulMatMullstm_2/lstm_cell_6/mul:z:0!lstm_2/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/MatMul_1MatMullstm_2/lstm_cell_6/mul_1:z:0!lstm_2/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/MatMul_2MatMullstm_2/lstm_cell_6/mul_2:z:0!lstm_2/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/MatMul_3MatMullstm_2/lstm_cell_6/mul_3:z:0!lstm_2/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????df
$lstm_2/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_2/lstm_cell_6/split_1/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_2/lstm_cell_6/split_1Split-lstm_2/lstm_cell_6/split_1/split_dim:output:01lstm_2/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
lstm_2/lstm_cell_6/BiasAddBiasAdd#lstm_2/lstm_cell_6/MatMul:product:0#lstm_2/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/BiasAdd_1BiasAdd%lstm_2/lstm_cell_6/MatMul_1:product:0#lstm_2/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/BiasAdd_2BiasAdd%lstm_2/lstm_cell_6/MatMul_2:product:0#lstm_2/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/BiasAdd_3BiasAdd%lstm_2/lstm_cell_6/MatMul_3:product:0#lstm_2/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_4Mullstm_2/zeros:output:0&lstm_2/lstm_cell_6/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_5Mullstm_2/zeros:output:0&lstm_2/lstm_cell_6/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_6Mullstm_2/zeros:output:0&lstm_2/lstm_cell_6/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_7Mullstm_2/zeros:output:0&lstm_2/lstm_cell_6/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
!lstm_2/lstm_cell_6/ReadVariableOpReadVariableOp*lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0w
&lstm_2/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_2/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   y
(lstm_2/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_2/lstm_cell_6/strided_sliceStridedSlice)lstm_2/lstm_cell_6/ReadVariableOp:value:0/lstm_2/lstm_cell_6/strided_slice/stack:output:01lstm_2/lstm_cell_6/strided_slice/stack_1:output:01lstm_2/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_2/lstm_cell_6/MatMul_4MatMullstm_2/lstm_cell_6/mul_4:z:0)lstm_2/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/addAddV2#lstm_2/lstm_cell_6/BiasAdd:output:0%lstm_2/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????ds
lstm_2/lstm_cell_6/SigmoidSigmoidlstm_2/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
#lstm_2/lstm_cell_6/ReadVariableOp_1ReadVariableOp*lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0y
(lstm_2/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   {
*lstm_2/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   {
*lstm_2/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_6/strided_slice_1StridedSlice+lstm_2/lstm_cell_6/ReadVariableOp_1:value:01lstm_2/lstm_cell_6/strided_slice_1/stack:output:03lstm_2/lstm_cell_6/strided_slice_1/stack_1:output:03lstm_2/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_2/lstm_cell_6/MatMul_5MatMullstm_2/lstm_cell_6/mul_5:z:0+lstm_2/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/add_1AddV2%lstm_2/lstm_cell_6/BiasAdd_1:output:0%lstm_2/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????dw
lstm_2/lstm_cell_6/Sigmoid_1Sigmoidlstm_2/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_8Mul lstm_2/lstm_cell_6/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
#lstm_2/lstm_cell_6/ReadVariableOp_2ReadVariableOp*lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0y
(lstm_2/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   {
*lstm_2/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  {
*lstm_2/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_6/strided_slice_2StridedSlice+lstm_2/lstm_cell_6/ReadVariableOp_2:value:01lstm_2/lstm_cell_6/strided_slice_2/stack:output:03lstm_2/lstm_cell_6/strided_slice_2/stack_1:output:03lstm_2/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_2/lstm_cell_6/MatMul_6MatMullstm_2/lstm_cell_6/mul_6:z:0+lstm_2/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/add_2AddV2%lstm_2/lstm_cell_6/BiasAdd_2:output:0%lstm_2/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????do
lstm_2/lstm_cell_6/TanhTanhlstm_2/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_9Mullstm_2/lstm_cell_6/Sigmoid:y:0lstm_2/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/add_3AddV2lstm_2/lstm_cell_6/mul_8:z:0lstm_2/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
#lstm_2/lstm_cell_6/ReadVariableOp_3ReadVariableOp*lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0y
(lstm_2/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  {
*lstm_2/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_2/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_6/strided_slice_3StridedSlice+lstm_2/lstm_cell_6/ReadVariableOp_3:value:01lstm_2/lstm_cell_6/strided_slice_3/stack:output:03lstm_2/lstm_cell_6/strided_slice_3/stack_1:output:03lstm_2/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_2/lstm_cell_6/MatMul_7MatMullstm_2/lstm_cell_6/mul_7:z:0+lstm_2/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/add_4AddV2%lstm_2/lstm_cell_6/BiasAdd_3:output:0%lstm_2/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????dw
lstm_2/lstm_cell_6/Sigmoid_2Sigmoidlstm_2/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????dq
lstm_2/lstm_cell_6/Tanh_1Tanhlstm_2/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_10Mul lstm_2/lstm_cell_6/Sigmoid_2:y:0lstm_2/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????du
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   e
#lstm_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0,lstm_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$lstm_2/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/TensorArrayV2_2TensorListReserve-lstm_2/TensorArrayV2_2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
>lstm_2/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0lstm_2/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_2/transpose_1:y:0Glstm_2/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:???o
lstm_2/zeros_like	ZerosLikelstm_2/lstm_cell_6/mul_10:z:0*
T0*'
_output_shapes
:?????????dj
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros_like:y:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_2/TensorArrayUnstack_1/TensorListFromTensor:output_handle:00lstm_2_lstm_cell_6_split_readvariableop_resource2lstm_2_lstm_cell_6_split_1_readvariableop_resource*lstm_2_lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *$
bodyR
lstm_2_while_body_607983*$
condR
lstm_2_while_cond_607982*`
output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *
parallel_iterations ?
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementso
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maskl
lstm_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose_2	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????db
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
dense_2/MatMulMatMullstm_2/strided_slice_3:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_1/dropout/MulMuldense_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????a
dropout_1/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup"^lstm_2/lstm_cell_6/ReadVariableOp$^lstm_2/lstm_cell_6/ReadVariableOp_1$^lstm_2/lstm_cell_6/ReadVariableOp_2$^lstm_2/lstm_cell_6/ReadVariableOp_3(^lstm_2/lstm_cell_6/split/ReadVariableOp*^lstm_2/lstm_cell_6/split_1/ReadVariableOp^lstm_2/while?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2F
!lstm_2/lstm_cell_6/ReadVariableOp!lstm_2/lstm_cell_6/ReadVariableOp2J
#lstm_2/lstm_cell_6/ReadVariableOp_1#lstm_2/lstm_cell_6/ReadVariableOp_12J
#lstm_2/lstm_cell_6/ReadVariableOp_2#lstm_2/lstm_cell_6/ReadVariableOp_22J
#lstm_2/lstm_cell_6/ReadVariableOp_3#lstm_2/lstm_cell_6/ReadVariableOp_32R
'lstm_2/lstm_cell_6/split/ReadVariableOp'lstm_2/lstm_cell_6/split/ReadVariableOp2V
)lstm_2/lstm_cell_6/split_1/ReadVariableOp)lstm_2/lstm_cell_6/split_1/ReadVariableOp2
lstm_2/whilelstm_2/while2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
G
__inference__creator_609928
identity: ??MutableHashTable|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
c
*__inference_dropout_1_layer_call_fn_609606

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_606496p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_606925

inputs
mask
<
)lstm_cell_6_split_readvariableop_resource:	@?:
+lstm_cell_6_split_1_readvariableop_resource:	?6
#lstm_cell_6_readvariableop_resource:	d?
identity??lstm_cell_6/ReadVariableOp?lstm_cell_6/ReadVariableOp_1?lstm_cell_6/ReadVariableOp_2?lstm_cell_6/ReadVariableOp_3? lstm_cell_6/split/ReadVariableOp?"lstm_cell_6/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????v

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskc
lstm_cell_6/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_likeFill$lstm_cell_6/ones_like/Shape:output:0$lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@^
lstm_cell_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout/MulMullstm_cell_6/ones_like:output:0"lstm_cell_6/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@g
lstm_cell_6/dropout/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_6/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_6/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0g
"lstm_cell_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_6/dropout/GreaterEqualGreaterEqual9lstm_cell_6/dropout/random_uniform/RandomUniform:output:0+lstm_cell_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout/CastCast$lstm_cell_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout/Mul_1Mullstm_cell_6/dropout/Mul:z:0lstm_cell_6/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_1/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_1/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_1/GreaterEqualGreaterEqual;lstm_cell_6/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_1/CastCast&lstm_cell_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_1/Mul_1Mullstm_cell_6/dropout_1/Mul:z:0lstm_cell_6/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_2/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_2/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_2/GreaterEqualGreaterEqual;lstm_cell_6/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_2/CastCast&lstm_cell_6/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_2/Mul_1Mullstm_cell_6/dropout_2/Mul:z:0lstm_cell_6/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_3/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_3/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_3/GreaterEqualGreaterEqual;lstm_cell_6/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_3/CastCast&lstm_cell_6/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_3/Mul_1Mullstm_cell_6/dropout_3/Mul:z:0lstm_cell_6/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@[
lstm_cell_6/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_like_1Fill&lstm_cell_6/ones_like_1/Shape:output:0&lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_4/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_4/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_4/GreaterEqualGreaterEqual;lstm_cell_6/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_4/CastCast&lstm_cell_6/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_4/Mul_1Mullstm_cell_6/dropout_4/Mul:z:0lstm_cell_6/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_5/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_5/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_5/GreaterEqualGreaterEqual;lstm_cell_6/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_5/CastCast&lstm_cell_6/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_5/Mul_1Mullstm_cell_6/dropout_5/Mul:z:0lstm_cell_6/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_6/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_6/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_6/GreaterEqualGreaterEqual;lstm_cell_6/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_6/CastCast&lstm_cell_6/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_6/Mul_1Mullstm_cell_6/dropout_6/Mul:z:0lstm_cell_6/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_7/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_7/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_7/GreaterEqualGreaterEqual;lstm_cell_6/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_7/CastCast&lstm_cell_6/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_7/Mul_1Mullstm_cell_6/dropout_7/Mul:z:0lstm_cell_6/dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/mulMulstrided_slice_2:output:0lstm_cell_6/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_1Mulstrided_slice_2:output:0lstm_cell_6/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_2Mulstrided_slice_2:output:0lstm_cell_6/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_3Mulstrided_slice_2:output:0lstm_cell_6/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_6/split/ReadVariableOpReadVariableOp)lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0(lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split
lstm_cell_6/MatMulMatMullstm_cell_6/mul:z:0lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_1MatMullstm_cell_6/mul_1:z:0lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_2MatMullstm_cell_6/mul_2:z:0lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_3MatMullstm_cell_6/mul_3:z:0lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????d_
lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_6/split_1/ReadVariableOpReadVariableOp+lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_6/split_1Split&lstm_cell_6/split_1/split_dim:output:0*lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
lstm_cell_6/BiasAddBiasAddlstm_cell_6/MatMul:product:0lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_1BiasAddlstm_cell_6/MatMul_1:product:0lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_2BiasAddlstm_cell_6/MatMul_2:product:0lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_3BiasAddlstm_cell_6/MatMul_3:product:0lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_4Mulzeros:output:0lstm_cell_6/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_5Mulzeros:output:0lstm_cell_6/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_6Mulzeros:output:0lstm_cell_6/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_7Mulzeros:output:0lstm_cell_6/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????d
lstm_cell_6/ReadVariableOpReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0p
lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   r
!lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_sliceStridedSlice"lstm_cell_6/ReadVariableOp:value:0(lstm_cell_6/strided_slice/stack:output:0*lstm_cell_6/strided_slice/stack_1:output:0*lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_4MatMullstm_cell_6/mul_4:z:0"lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/addAddV2lstm_cell_6/BiasAdd:output:0lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????de
lstm_cell_6/SigmoidSigmoidlstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_1ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   t
#lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_1StridedSlice$lstm_cell_6/ReadVariableOp_1:value:0*lstm_cell_6/strided_slice_1/stack:output:0,lstm_cell_6/strided_slice_1/stack_1:output:0,lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_5MatMullstm_cell_6/mul_5:z:0$lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_1AddV2lstm_cell_6/BiasAdd_1:output:0lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????dw
lstm_cell_6/mul_8Mullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_2ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_2StridedSlice$lstm_cell_6/ReadVariableOp_2:value:0*lstm_cell_6/strided_slice_2/stack:output:0,lstm_cell_6/strided_slice_2/stack_1:output:0,lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_6MatMullstm_cell_6/mul_6:z:0$lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_2AddV2lstm_cell_6/BiasAdd_2:output:0lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????da
lstm_cell_6/TanhTanhlstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????dy
lstm_cell_6/mul_9Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????dz
lstm_cell_6/add_3AddV2lstm_cell_6/mul_8:z:0lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_3ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_3StridedSlice$lstm_cell_6/ReadVariableOp_3:value:0*lstm_cell_6/strided_slice_3/stack:output:0,lstm_cell_6/strided_slice_3/stack_1:output:0,lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_7MatMullstm_cell_6/mul_7:z:0$lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_4AddV2lstm_cell_6/BiasAdd_3:output:0lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????dc
lstm_cell_6/Tanh_1Tanhlstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d~
lstm_cell_6/mul_10Mullstm_cell_6/Sigmoid_2:y:0lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:???a

zeros_like	ZerosLikelstm_cell_6/mul_10:z:0*
T0*'
_output_shapes
:?????????dc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_6_split_readvariableop_resource+lstm_cell_6_split_1_readvariableop_resource#lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *
bodyR
while_body_606708*
condR
while_cond_606707*`
output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^lstm_cell_6/ReadVariableOp^lstm_cell_6/ReadVariableOp_1^lstm_cell_6/ReadVariableOp_2^lstm_cell_6/ReadVariableOp_3!^lstm_cell_6/split/ReadVariableOp#^lstm_cell_6/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????????????@:??????????????????: : : 28
lstm_cell_6/ReadVariableOplstm_cell_6/ReadVariableOp2<
lstm_cell_6/ReadVariableOp_1lstm_cell_6/ReadVariableOp_12<
lstm_cell_6/ReadVariableOp_2lstm_cell_6/ReadVariableOp_22<
lstm_cell_6/ReadVariableOp_3lstm_cell_6/ReadVariableOp_32D
 lstm_cell_6/split/ReadVariableOp lstm_cell_6/split/ReadVariableOp2H
"lstm_cell_6/split_1/ReadVariableOp"lstm_cell_6/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
-__inference_sequential_1_layer_call_fn_607401

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?@
	unknown_4:	@?
	unknown_5:	?
	unknown_6:	d?
	unknown_7:	d?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_607049o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
while_cond_606235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_14
0while_while_cond_606235___redundant_placeholder04
0while_while_cond_606235___redundant_placeholder14
0while_while_cond_606235___redundant_placeholder24
0while_while_cond_606235___redundant_placeholder34
0while_while_cond_606235___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :?????????d:?????????d:?????????d: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
?
lstm_2_while_cond_607575*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3
lstm_2_while_placeholder_4,
(lstm_2_while_less_lstm_2_strided_slice_1B
>lstm_2_while_lstm_2_while_cond_607575___redundant_placeholder0B
>lstm_2_while_lstm_2_while_cond_607575___redundant_placeholder1B
>lstm_2_while_lstm_2_while_cond_607575___redundant_placeholder2B
>lstm_2_while_lstm_2_while_cond_607575___redundant_placeholder3B
>lstm_2_while_lstm_2_while_cond_607575___redundant_placeholder4
lstm_2_while_identity
~
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: Y
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :?????????d:?????????d:?????????d: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
?
-__inference_sequential_1_layer_call_fn_606466
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?@
	unknown_4:	@?
	unknown_5:	?
	unknown_6:	d?
	unknown_7:	d?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_606439o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?	
while_body_608703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_6_split_readvariableop_resource_0:	@?B
3while_lstm_cell_6_split_1_readvariableop_resource_0:	?>
+while_lstm_cell_6_readvariableop_resource_0:	d?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_6_split_readvariableop_resource:	@?@
1while_lstm_cell_6_split_1_readvariableop_resource:	?<
)while_lstm_cell_6_readvariableop_resource:	d??? while/lstm_cell_6/ReadVariableOp?"while/lstm_cell_6/ReadVariableOp_1?"while/lstm_cell_6/ReadVariableOp_2?"while/lstm_cell_6/ReadVariableOp_3?&while/lstm_cell_6/split/ReadVariableOp?(while/lstm_cell_6/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
!while/lstm_cell_6/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_likeFill*while/lstm_cell_6/ones_like/Shape:output:0*while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@d
while/lstm_cell_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout/MulMul$while/lstm_cell_6/ones_like:output:0(while/lstm_cell_6/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@s
while/lstm_cell_6/dropout/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_6/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_6/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0m
(while/lstm_cell_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_6/dropout/GreaterEqualGreaterEqual?while/lstm_cell_6/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/dropout/CastCast*while/lstm_cell_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
while/lstm_cell_6/dropout/Mul_1Mul!while/lstm_cell_6/dropout/Mul:z:0"while/lstm_cell_6/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_1/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_1/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_1/CastCast,while/lstm_cell_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_1/Mul_1Mul#while/lstm_cell_6/dropout_1/Mul:z:0$while/lstm_cell_6/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_2/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_2/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_2/CastCast,while/lstm_cell_6/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_2/Mul_1Mul#while/lstm_cell_6/dropout_2/Mul:z:0$while/lstm_cell_6/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_3/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_3/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_3/CastCast,while/lstm_cell_6/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_3/Mul_1Mul#while/lstm_cell_6/dropout_3/Mul:z:0$while/lstm_cell_6/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@f
#while/lstm_cell_6/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:h
#while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_like_1Fill,while/lstm_cell_6/ones_like_1/Shape:output:0,while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_4/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_4/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_4/CastCast,while/lstm_cell_6/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_4/Mul_1Mul#while/lstm_cell_6/dropout_4/Mul:z:0$while/lstm_cell_6/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_5/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_5/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_5/CastCast,while/lstm_cell_6/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_5/Mul_1Mul#while/lstm_cell_6/dropout_5/Mul:z:0$while/lstm_cell_6/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_6/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_6/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_6/CastCast,while/lstm_cell_6/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_6/Mul_1Mul#while/lstm_cell_6/dropout_6/Mul:z:0$while/lstm_cell_6/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_7/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_7/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_7/CastCast,while/lstm_cell_6/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_7/Mul_1Mul#while/lstm_cell_6/dropout_7/Mul:z:0$while/lstm_cell_6/dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_6/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_6/split/ReadVariableOpReadVariableOp1while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0.while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
while/lstm_cell_6/MatMulMatMulwhile/lstm_cell_6/mul:z:0 while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_1MatMulwhile/lstm_cell_6/mul_1:z:0 while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_2MatMulwhile/lstm_cell_6/mul_2:z:0 while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_3MatMulwhile/lstm_cell_6/mul_3:z:0 while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????de
#while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_6/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_6/split_1Split,while/lstm_cell_6/split_1/split_dim:output:00while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
while/lstm_cell_6/BiasAddBiasAdd"while/lstm_cell_6/MatMul:product:0"while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_1BiasAdd$while/lstm_cell_6/MatMul_1:product:0"while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_2BiasAdd$while/lstm_cell_6/MatMul_2:product:0"while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_3BiasAdd$while/lstm_cell_6/MatMul_3:product:0"while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_4Mulwhile_placeholder_2%while/lstm_cell_6/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_5Mulwhile_placeholder_2%while/lstm_cell_6/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_6Mulwhile_placeholder_2%while/lstm_cell_6/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_7Mulwhile_placeholder_2%while/lstm_cell_6/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/ReadVariableOpReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0v
%while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   x
'while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_6/strided_sliceStridedSlice(while/lstm_cell_6/ReadVariableOp:value:0.while/lstm_cell_6/strided_slice/stack:output:00while/lstm_cell_6/strided_slice/stack_1:output:00while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_4MatMulwhile/lstm_cell_6/mul_4:z:0(while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/addAddV2"while/lstm_cell_6/BiasAdd:output:0$while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????dq
while/lstm_cell_6/SigmoidSigmoidwhile/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_1ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   z
)while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_1StridedSlice*while/lstm_cell_6/ReadVariableOp_1:value:00while/lstm_cell_6/strided_slice_1/stack:output:02while/lstm_cell_6/strided_slice_1/stack_1:output:02while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_5MatMulwhile/lstm_cell_6/mul_5:z:0*while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_1AddV2$while/lstm_cell_6/BiasAdd_1:output:0$while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_1Sigmoidwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_8Mulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_2ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_2StridedSlice*while/lstm_cell_6/ReadVariableOp_2:value:00while/lstm_cell_6/strided_slice_2/stack:output:02while/lstm_cell_6/strided_slice_2/stack_1:output:02while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_6MatMulwhile/lstm_cell_6/mul_6:z:0*while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_2AddV2$while/lstm_cell_6/BiasAdd_2:output:0$while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????dm
while/lstm_cell_6/TanhTanhwhile/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_9Mulwhile/lstm_cell_6/Sigmoid:y:0while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_3AddV2while/lstm_cell_6/mul_8:z:0while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_3ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_3StridedSlice*while/lstm_cell_6/ReadVariableOp_3:value:00while/lstm_cell_6/strided_slice_3/stack:output:02while/lstm_cell_6/strided_slice_3/stack_1:output:02while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_7MatMulwhile/lstm_cell_6/mul_7:z:0*while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_4AddV2$while/lstm_cell_6/BiasAdd_3:output:0$while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_2Sigmoidwhile/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????do
while/lstm_cell_6/Tanh_1Tanhwhile/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_10Mulwhile/lstm_cell_6/Sigmoid_2:y:0while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_6/mul_10:z:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_6/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:?????????dx
while/Identity_5Identitywhile/lstm_cell_6/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp!^while/lstm_cell_6/ReadVariableOp#^while/lstm_cell_6/ReadVariableOp_1#^while/lstm_cell_6/ReadVariableOp_2#^while/lstm_cell_6/ReadVariableOp_3'^while/lstm_cell_6/split/ReadVariableOp)^while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"X
)while_lstm_cell_6_readvariableop_resource+while_lstm_cell_6_readvariableop_resource_0"h
1while_lstm_cell_6_split_1_readvariableop_resource3while_lstm_cell_6_split_1_readvariableop_resource_0"d
/while_lstm_cell_6_split_readvariableop_resource1while_lstm_cell_6_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????d:?????????d: : : : : 2D
 while/lstm_cell_6/ReadVariableOp while/lstm_cell_6/ReadVariableOp2H
"while/lstm_cell_6/ReadVariableOp_1"while/lstm_cell_6/ReadVariableOp_12H
"while/lstm_cell_6/ReadVariableOp_2"while/lstm_cell_6/ReadVariableOp_22H
"while/lstm_cell_6/ReadVariableOp_3"while/lstm_cell_6/ReadVariableOp_32P
&while/lstm_cell_6/split/ReadVariableOp&while/lstm_cell_6/split/ReadVariableOp2T
(while/lstm_cell_6/split_1/ReadVariableOp(while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_606432

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
'__inference_lstm_2_layer_call_fn_608272

inputs
mask

unknown:	@?
	unknown_0:	?
	unknown_1:	d?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_606389o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????????????@:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
-
__inference__destroyer_609938
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
while_cond_608393
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_608393___redundant_placeholder04
0while_while_cond_608393___redundant_placeholder14
0while_while_cond_608393___redundant_placeholder24
0while_while_cond_608393___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
??
?

while_body_609022
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0D
1while_lstm_cell_6_split_readvariableop_resource_0:	@?B
3while_lstm_cell_6_split_1_readvariableop_resource_0:	?>
+while_lstm_cell_6_readvariableop_resource_0:	d?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorB
/while_lstm_cell_6_split_readvariableop_resource:	@?@
1while_lstm_cell_6_split_1_readvariableop_resource:	?<
)while_lstm_cell_6_readvariableop_resource:	d??? while/lstm_cell_6/ReadVariableOp?"while/lstm_cell_6/ReadVariableOp_1?"while/lstm_cell_6/ReadVariableOp_2?"while/lstm_cell_6/ReadVariableOp_3?&while/lstm_cell_6/split/ReadVariableOp?(while/lstm_cell_6/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
!while/lstm_cell_6/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_likeFill*while/lstm_cell_6/ones_like/Shape:output:0*while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@f
#while/lstm_cell_6/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:h
#while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_like_1Fill,while/lstm_cell_6/ones_like_1/Shape:output:0,while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_6/split/ReadVariableOpReadVariableOp1while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0.while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
while/lstm_cell_6/MatMulMatMulwhile/lstm_cell_6/mul:z:0 while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_1MatMulwhile/lstm_cell_6/mul_1:z:0 while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_2MatMulwhile/lstm_cell_6/mul_2:z:0 while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_3MatMulwhile/lstm_cell_6/mul_3:z:0 while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????de
#while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_6/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_6/split_1Split,while/lstm_cell_6/split_1/split_dim:output:00while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
while/lstm_cell_6/BiasAddBiasAdd"while/lstm_cell_6/MatMul:product:0"while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_1BiasAdd$while/lstm_cell_6/MatMul_1:product:0"while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_2BiasAdd$while/lstm_cell_6/MatMul_2:product:0"while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_3BiasAdd$while/lstm_cell_6/MatMul_3:product:0"while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_4Mulwhile_placeholder_3&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_5Mulwhile_placeholder_3&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_6Mulwhile_placeholder_3&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_7Mulwhile_placeholder_3&while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/ReadVariableOpReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0v
%while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   x
'while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_6/strided_sliceStridedSlice(while/lstm_cell_6/ReadVariableOp:value:0.while/lstm_cell_6/strided_slice/stack:output:00while/lstm_cell_6/strided_slice/stack_1:output:00while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_4MatMulwhile/lstm_cell_6/mul_4:z:0(while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/addAddV2"while/lstm_cell_6/BiasAdd:output:0$while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????dq
while/lstm_cell_6/SigmoidSigmoidwhile/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_1ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   z
)while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_1StridedSlice*while/lstm_cell_6/ReadVariableOp_1:value:00while/lstm_cell_6/strided_slice_1/stack:output:02while/lstm_cell_6/strided_slice_1/stack_1:output:02while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_5MatMulwhile/lstm_cell_6/mul_5:z:0*while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_1AddV2$while/lstm_cell_6/BiasAdd_1:output:0$while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_1Sigmoidwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_8Mulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_2ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_2StridedSlice*while/lstm_cell_6/ReadVariableOp_2:value:00while/lstm_cell_6/strided_slice_2/stack:output:02while/lstm_cell_6/strided_slice_2/stack_1:output:02while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_6MatMulwhile/lstm_cell_6/mul_6:z:0*while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_2AddV2$while/lstm_cell_6/BiasAdd_2:output:0$while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????dm
while/lstm_cell_6/TanhTanhwhile/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_9Mulwhile/lstm_cell_6/Sigmoid:y:0while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_3AddV2while/lstm_cell_6/mul_8:z:0while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_3ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_3StridedSlice*while/lstm_cell_6/ReadVariableOp_3:value:00while/lstm_cell_6/strided_slice_3/stack:output:02while/lstm_cell_6/strided_slice_3/stack_1:output:02while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_7MatMulwhile/lstm_cell_6/mul_7:z:0*while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_4AddV2$while/lstm_cell_6/BiasAdd_3:output:0$while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_2Sigmoidwhile/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????do
while/lstm_cell_6/Tanh_1Tanhwhile/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_10Mulwhile/lstm_cell_6/Sigmoid_2:y:0while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????de
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_6/mul_10:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????dg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????g
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:??????????
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_6/mul_10:z:0while_placeholder_3*
T0*'
_output_shapes
:?????????d?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_6/add_3:z:0while_placeholder_4*
T0*'
_output_shapes
:?????????dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*'
_output_shapes
:?????????dv
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*'
_output_shapes
:?????????dv
while/Identity_6Identitywhile/SelectV2_2:output:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp!^while/lstm_cell_6/ReadVariableOp#^while/lstm_cell_6/ReadVariableOp_1#^while/lstm_cell_6/ReadVariableOp_2#^while/lstm_cell_6/ReadVariableOp_3'^while/lstm_cell_6/split/ReadVariableOp)^while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_6_readvariableop_resource+while_lstm_cell_6_readvariableop_resource_0"h
1while_lstm_cell_6_split_1_readvariableop_resource3while_lstm_cell_6_split_1_readvariableop_resource_0"d
/while_lstm_cell_6_split_readvariableop_resource1while_lstm_cell_6_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : 2D
 while/lstm_cell_6/ReadVariableOp while/lstm_cell_6/ReadVariableOp2H
"while/lstm_cell_6/ReadVariableOp_1"while/lstm_cell_6/ReadVariableOp_12H
"while/lstm_cell_6/ReadVariableOp_2"while/lstm_cell_6/ReadVariableOp_22H
"while/lstm_cell_6/ReadVariableOp_3"while/lstm_cell_6/ReadVariableOp_32P
&while/lstm_cell_6/split/ReadVariableOp&while/lstm_cell_6/split/ReadVariableOp2T
(while/lstm_cell_6/split_1/ReadVariableOp(while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
-__inference_sequential_1_layer_call_fn_607372

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?@
	unknown_4:	@?
	unknown_5:	?
	unknown_6:	d?
	unknown_7:	d?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_606439o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
;
__inference__creator_609910
identity??
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name8292*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
??
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_609576

inputs
mask
<
)lstm_cell_6_split_readvariableop_resource:	@?:
+lstm_cell_6_split_1_readvariableop_resource:	?6
#lstm_cell_6_readvariableop_resource:	d?
identity??lstm_cell_6/ReadVariableOp?lstm_cell_6/ReadVariableOp_1?lstm_cell_6/ReadVariableOp_2?lstm_cell_6/ReadVariableOp_3? lstm_cell_6/split/ReadVariableOp?"lstm_cell_6/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????v

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskc
lstm_cell_6/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_likeFill$lstm_cell_6/ones_like/Shape:output:0$lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@^
lstm_cell_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout/MulMullstm_cell_6/ones_like:output:0"lstm_cell_6/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@g
lstm_cell_6/dropout/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_6/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_6/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0g
"lstm_cell_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_6/dropout/GreaterEqualGreaterEqual9lstm_cell_6/dropout/random_uniform/RandomUniform:output:0+lstm_cell_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout/CastCast$lstm_cell_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout/Mul_1Mullstm_cell_6/dropout/Mul:z:0lstm_cell_6/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_1/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_1/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_1/GreaterEqualGreaterEqual;lstm_cell_6/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_1/CastCast&lstm_cell_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_1/Mul_1Mullstm_cell_6/dropout_1/Mul:z:0lstm_cell_6/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_2/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_2/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_2/GreaterEqualGreaterEqual;lstm_cell_6/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_2/CastCast&lstm_cell_6/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_2/Mul_1Mullstm_cell_6/dropout_2/Mul:z:0lstm_cell_6/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_3/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_3/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_3/GreaterEqualGreaterEqual;lstm_cell_6/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_3/CastCast&lstm_cell_6/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_3/Mul_1Mullstm_cell_6/dropout_3/Mul:z:0lstm_cell_6/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@[
lstm_cell_6/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_like_1Fill&lstm_cell_6/ones_like_1/Shape:output:0&lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_4/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_4/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_4/GreaterEqualGreaterEqual;lstm_cell_6/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_4/CastCast&lstm_cell_6/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_4/Mul_1Mullstm_cell_6/dropout_4/Mul:z:0lstm_cell_6/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_5/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_5/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_5/GreaterEqualGreaterEqual;lstm_cell_6/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_5/CastCast&lstm_cell_6/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_5/Mul_1Mullstm_cell_6/dropout_5/Mul:z:0lstm_cell_6/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_6/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_6/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_6/GreaterEqualGreaterEqual;lstm_cell_6/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_6/CastCast&lstm_cell_6/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_6/Mul_1Mullstm_cell_6/dropout_6/Mul:z:0lstm_cell_6/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_7/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_7/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_7/GreaterEqualGreaterEqual;lstm_cell_6/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_7/CastCast&lstm_cell_6/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_7/Mul_1Mullstm_cell_6/dropout_7/Mul:z:0lstm_cell_6/dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/mulMulstrided_slice_2:output:0lstm_cell_6/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_1Mulstrided_slice_2:output:0lstm_cell_6/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_2Mulstrided_slice_2:output:0lstm_cell_6/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_3Mulstrided_slice_2:output:0lstm_cell_6/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_6/split/ReadVariableOpReadVariableOp)lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0(lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split
lstm_cell_6/MatMulMatMullstm_cell_6/mul:z:0lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_1MatMullstm_cell_6/mul_1:z:0lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_2MatMullstm_cell_6/mul_2:z:0lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_3MatMullstm_cell_6/mul_3:z:0lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????d_
lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_6/split_1/ReadVariableOpReadVariableOp+lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_6/split_1Split&lstm_cell_6/split_1/split_dim:output:0*lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
lstm_cell_6/BiasAddBiasAddlstm_cell_6/MatMul:product:0lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_1BiasAddlstm_cell_6/MatMul_1:product:0lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_2BiasAddlstm_cell_6/MatMul_2:product:0lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_3BiasAddlstm_cell_6/MatMul_3:product:0lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_4Mulzeros:output:0lstm_cell_6/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_5Mulzeros:output:0lstm_cell_6/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_6Mulzeros:output:0lstm_cell_6/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_7Mulzeros:output:0lstm_cell_6/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????d
lstm_cell_6/ReadVariableOpReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0p
lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   r
!lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_sliceStridedSlice"lstm_cell_6/ReadVariableOp:value:0(lstm_cell_6/strided_slice/stack:output:0*lstm_cell_6/strided_slice/stack_1:output:0*lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_4MatMullstm_cell_6/mul_4:z:0"lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/addAddV2lstm_cell_6/BiasAdd:output:0lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????de
lstm_cell_6/SigmoidSigmoidlstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_1ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   t
#lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_1StridedSlice$lstm_cell_6/ReadVariableOp_1:value:0*lstm_cell_6/strided_slice_1/stack:output:0,lstm_cell_6/strided_slice_1/stack_1:output:0,lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_5MatMullstm_cell_6/mul_5:z:0$lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_1AddV2lstm_cell_6/BiasAdd_1:output:0lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????dw
lstm_cell_6/mul_8Mullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_2ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_2StridedSlice$lstm_cell_6/ReadVariableOp_2:value:0*lstm_cell_6/strided_slice_2/stack:output:0,lstm_cell_6/strided_slice_2/stack_1:output:0,lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_6MatMullstm_cell_6/mul_6:z:0$lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_2AddV2lstm_cell_6/BiasAdd_2:output:0lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????da
lstm_cell_6/TanhTanhlstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????dy
lstm_cell_6/mul_9Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????dz
lstm_cell_6/add_3AddV2lstm_cell_6/mul_8:z:0lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_3ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_3StridedSlice$lstm_cell_6/ReadVariableOp_3:value:0*lstm_cell_6/strided_slice_3/stack:output:0,lstm_cell_6/strided_slice_3/stack_1:output:0,lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_7MatMullstm_cell_6/mul_7:z:0$lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_4AddV2lstm_cell_6/BiasAdd_3:output:0lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????dc
lstm_cell_6/Tanh_1Tanhlstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d~
lstm_cell_6/mul_10Mullstm_cell_6/Sigmoid_2:y:0lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:???a

zeros_like	ZerosLikelstm_cell_6/mul_10:z:0*
T0*'
_output_shapes
:?????????dc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_6_split_readvariableop_resource+lstm_cell_6_split_1_readvariableop_resource#lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *
bodyR
while_body_609359*
condR
while_cond_609358*`
output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^lstm_cell_6/ReadVariableOp^lstm_cell_6/ReadVariableOp_1^lstm_cell_6/ReadVariableOp_2^lstm_cell_6/ReadVariableOp_3!^lstm_cell_6/split/ReadVariableOp#^lstm_cell_6/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????????????@:??????????????????: : : 28
lstm_cell_6/ReadVariableOplstm_cell_6/ReadVariableOp2<
lstm_cell_6/ReadVariableOp_1lstm_cell_6/ReadVariableOp_12<
lstm_cell_6/ReadVariableOp_2lstm_cell_6/ReadVariableOp_22<
lstm_cell_6/ReadVariableOp_3lstm_cell_6/ReadVariableOp_32D
 lstm_cell_6/split/ReadVariableOp lstm_cell_6/split/ReadVariableOp2H
"lstm_cell_6/split_1/ReadVariableOp"lstm_cell_6/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?
?
(__inference_dense_3_layer_call_fn_609632

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_606432o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ۊ
?
"__inference__traced_restore_610223
file_prefix:
'assignvariableop_embedding_1_embeddings:	?@4
!assignvariableop_1_dense_2_kernel:	d?.
assignvariableop_2_dense_2_bias:	?4
!assignvariableop_3_dense_3_kernel:	?-
assignvariableop_4_dense_3_bias:?
,assignvariableop_5_lstm_2_lstm_cell_6_kernel:	@?I
6assignvariableop_6_lstm_2_lstm_cell_6_recurrent_kernel:	d?9
*assignvariableop_7_lstm_2_lstm_cell_6_bias:	?&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: 
mutablehashtable: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: D
1assignvariableop_17_adam_embedding_1_embeddings_m:	?@<
)assignvariableop_18_adam_dense_2_kernel_m:	d?6
'assignvariableop_19_adam_dense_2_bias_m:	?<
)assignvariableop_20_adam_dense_3_kernel_m:	?5
'assignvariableop_21_adam_dense_3_bias_m:G
4assignvariableop_22_adam_lstm_2_lstm_cell_6_kernel_m:	@?Q
>assignvariableop_23_adam_lstm_2_lstm_cell_6_recurrent_kernel_m:	d?A
2assignvariableop_24_adam_lstm_2_lstm_cell_6_bias_m:	?D
1assignvariableop_25_adam_embedding_1_embeddings_v:	?@<
)assignvariableop_26_adam_dense_2_kernel_v:	d?6
'assignvariableop_27_adam_dense_2_bias_v:	?<
)assignvariableop_28_adam_dense_3_kernel_v:	?5
'assignvariableop_29_adam_dense_3_bias_v:G
4assignvariableop_30_adam_lstm_2_lstm_cell_6_kernel_v:	@?Q
>assignvariableop_31_adam_lstm_2_lstm_cell_6_recurrent_kernel_v:	d?A
2assignvariableop_32_adam_lstm_2_lstm_cell_6_bias_v:	?
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?StatefulPartitionedCall?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_3_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_lstm_2_lstm_cell_6_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp6assignvariableop_6_lstm_2_lstm_cell_6_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_lstm_2_lstm_cell_6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
StatefulPartitionedCallStatefulPartitionedCallmutablehashtableRestoreV2:tensors:13RestoreV2:tensors:14"/device:CPU:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_restore_from_tensors_610178_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_embedding_1_embeddings_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_2_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_2_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_3_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_3_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_2_lstm_cell_6_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_lstm_2_lstm_cell_6_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_lstm_2_lstm_cell_6_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_embedding_1_embeddings_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_2_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_2_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_3_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_3_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_2_lstm_cell_6_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_lstm_2_lstm_cell_6_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_lstm_2_lstm_cell_6_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp^StatefulPartitionedCall"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_922
StatefulPartitionedCallStatefulPartitionedCall:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?#
?
while_body_605660
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_6_605684_0:	@?)
while_lstm_cell_6_605686_0:	?-
while_lstm_cell_6_605688_0:	d?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_6_605684:	@?'
while_lstm_cell_6_605686:	?+
while_lstm_cell_6_605688:	d???)while/lstm_cell_6/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
)while/lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_6_605684_0while_lstm_cell_6_605686_0while_lstm_cell_6_605688_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_605645r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_6/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_6/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?
while/Identity_5Identity2while/lstm_cell_6/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????dx

while/NoOpNoOp*^while/lstm_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_6_605684while_lstm_cell_6_605684_0"6
while_lstm_cell_6_605686while_lstm_cell_6_605686_0"6
while_lstm_cell_6_605688while_lstm_cell_6_605688_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????d:?????????d: : : : : 2V
)while/lstm_cell_6/StatefulPartitionedCall)while/lstm_cell_6/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?D
?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_605645

inputs

states
states_10
split_readvariableop_resource:	@?.
split_1_readvariableop_resource:	?*
readvariableop_resource:	d?
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????dX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:?????????@Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????@Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????@Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:?????????@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:?????????d_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:?????????d_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:?????????d_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:?????????dS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????dl
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????dl
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????dl
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????d\
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????d\
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????d\
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????d\
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????dg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????di
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????dh
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????dW
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????di
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????dh
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????dU
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????dV
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:?????????di
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????dh
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????dQ
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????dK
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????dZ
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????d[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????dZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:?????????d:?????????d: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
??
?

while_body_609359
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0D
1while_lstm_cell_6_split_readvariableop_resource_0:	@?B
3while_lstm_cell_6_split_1_readvariableop_resource_0:	?>
+while_lstm_cell_6_readvariableop_resource_0:	d?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorB
/while_lstm_cell_6_split_readvariableop_resource:	@?@
1while_lstm_cell_6_split_1_readvariableop_resource:	?<
)while_lstm_cell_6_readvariableop_resource:	d??? while/lstm_cell_6/ReadVariableOp?"while/lstm_cell_6/ReadVariableOp_1?"while/lstm_cell_6/ReadVariableOp_2?"while/lstm_cell_6/ReadVariableOp_3?&while/lstm_cell_6/split/ReadVariableOp?(while/lstm_cell_6/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
!while/lstm_cell_6/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:f
!while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_likeFill*while/lstm_cell_6/ones_like/Shape:output:0*while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@d
while/lstm_cell_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout/MulMul$while/lstm_cell_6/ones_like:output:0(while/lstm_cell_6/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@s
while/lstm_cell_6/dropout/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
6while/lstm_cell_6/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_6/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0m
(while/lstm_cell_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
&while/lstm_cell_6/dropout/GreaterEqualGreaterEqual?while/lstm_cell_6/dropout/random_uniform/RandomUniform:output:01while/lstm_cell_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/dropout/CastCast*while/lstm_cell_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
while/lstm_cell_6/dropout/Mul_1Mul!while/lstm_cell_6/dropout/Mul:z:0"while/lstm_cell_6/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_1/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_1/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_1/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_1/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_1/CastCast,while/lstm_cell_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_1/Mul_1Mul#while/lstm_cell_6/dropout_1/Mul:z:0$while/lstm_cell_6/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_2/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_2/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_2/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_2/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_2/CastCast,while/lstm_cell_6/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_2/Mul_1Mul#while/lstm_cell_6/dropout_2/Mul:z:0$while/lstm_cell_6/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@f
!while/lstm_cell_6/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_3/MulMul$while/lstm_cell_6/ones_like:output:0*while/lstm_cell_6/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@u
!while/lstm_cell_6/dropout_3/ShapeShape$while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0o
*while/lstm_cell_6/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_3/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_3/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
 while/lstm_cell_6/dropout_3/CastCast,while/lstm_cell_6/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
!while/lstm_cell_6/dropout_3/Mul_1Mul#while/lstm_cell_6/dropout_3/Mul:z:0$while/lstm_cell_6/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@f
#while/lstm_cell_6/ones_like_1/ShapeShapewhile_placeholder_3*
T0*
_output_shapes
:h
#while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/ones_like_1Fill,while/lstm_cell_6/ones_like_1/Shape:output:0,while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_4/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_4/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_4/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_4/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_4/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_4/CastCast,while/lstm_cell_6/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_4/Mul_1Mul#while/lstm_cell_6/dropout_4/Mul:z:0$while/lstm_cell_6/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_5/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_5/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_5/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_5/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_5/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_5/CastCast,while/lstm_cell_6/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_5/Mul_1Mul#while/lstm_cell_6/dropout_5/Mul:z:0$while/lstm_cell_6/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_6/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_6/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_6/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_6/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_6/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_6/CastCast,while/lstm_cell_6/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_6/Mul_1Mul#while/lstm_cell_6/dropout_6/Mul:z:0$while/lstm_cell_6/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????df
!while/lstm_cell_6/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
while/lstm_cell_6/dropout_7/MulMul&while/lstm_cell_6/ones_like_1:output:0*while/lstm_cell_6/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dw
!while/lstm_cell_6/dropout_7/ShapeShape&while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
8while/lstm_cell_6/dropout_7/random_uniform/RandomUniformRandomUniform*while/lstm_cell_6/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0o
*while/lstm_cell_6/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
(while/lstm_cell_6/dropout_7/GreaterEqualGreaterEqualAwhile/lstm_cell_6/dropout_7/random_uniform/RandomUniform:output:03while/lstm_cell_6/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/dropout_7/CastCast,while/lstm_cell_6/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
!while/lstm_cell_6/dropout_7/Mul_1Mul#while/lstm_cell_6/dropout_7/Mul:z:0$while/lstm_cell_6/dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell_6/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
while/lstm_cell_6/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_6/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@c
!while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
&while/lstm_cell_6/split/ReadVariableOpReadVariableOp1while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
while/lstm_cell_6/splitSplit*while/lstm_cell_6/split/split_dim:output:0.while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
while/lstm_cell_6/MatMulMatMulwhile/lstm_cell_6/mul:z:0 while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_1MatMulwhile/lstm_cell_6/mul_1:z:0 while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_2MatMulwhile/lstm_cell_6/mul_2:z:0 while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/MatMul_3MatMulwhile/lstm_cell_6/mul_3:z:0 while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????de
#while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(while/lstm_cell_6/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
while/lstm_cell_6/split_1Split,while/lstm_cell_6/split_1/split_dim:output:00while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
while/lstm_cell_6/BiasAddBiasAdd"while/lstm_cell_6/MatMul:product:0"while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_1BiasAdd$while/lstm_cell_6/MatMul_1:product:0"while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_2BiasAdd$while/lstm_cell_6/MatMul_2:product:0"while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/BiasAdd_3BiasAdd$while/lstm_cell_6/MatMul_3:product:0"while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_4Mulwhile_placeholder_3%while/lstm_cell_6/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_5Mulwhile_placeholder_3%while/lstm_cell_6/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_6Mulwhile_placeholder_3%while/lstm_cell_6/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_7Mulwhile_placeholder_3%while/lstm_cell_6/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
 while/lstm_cell_6/ReadVariableOpReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0v
%while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   x
'while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
while/lstm_cell_6/strided_sliceStridedSlice(while/lstm_cell_6/ReadVariableOp:value:0.while/lstm_cell_6/strided_slice/stack:output:00while/lstm_cell_6/strided_slice/stack_1:output:00while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_4MatMulwhile/lstm_cell_6/mul_4:z:0(while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/addAddV2"while/lstm_cell_6/BiasAdd:output:0$while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????dq
while/lstm_cell_6/SigmoidSigmoidwhile/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_1ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   z
)while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_1StridedSlice*while/lstm_cell_6/ReadVariableOp_1:value:00while/lstm_cell_6/strided_slice_1/stack:output:02while/lstm_cell_6/strided_slice_1/stack_1:output:02while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_5MatMulwhile/lstm_cell_6/mul_5:z:0*while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_1AddV2$while/lstm_cell_6/BiasAdd_1:output:0$while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_1Sigmoidwhile/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_8Mulwhile/lstm_cell_6/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_2ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   z
)while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_2StridedSlice*while/lstm_cell_6/ReadVariableOp_2:value:00while/lstm_cell_6/strided_slice_2/stack:output:02while/lstm_cell_6/strided_slice_2/stack_1:output:02while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_6MatMulwhile/lstm_cell_6/mul_6:z:0*while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_2AddV2$while/lstm_cell_6/BiasAdd_2:output:0$while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????dm
while/lstm_cell_6/TanhTanhwhile/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_9Mulwhile/lstm_cell_6/Sigmoid:y:0while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_3AddV2while/lstm_cell_6/mul_8:z:0while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
"while/lstm_cell_6/ReadVariableOp_3ReadVariableOp+while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0x
'while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  z
)while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        z
)while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
!while/lstm_cell_6/strided_slice_3StridedSlice*while/lstm_cell_6/ReadVariableOp_3:value:00while/lstm_cell_6/strided_slice_3/stack:output:02while/lstm_cell_6/strided_slice_3/stack_1:output:02while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
while/lstm_cell_6/MatMul_7MatMulwhile/lstm_cell_6/mul_7:z:0*while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/add_4AddV2$while/lstm_cell_6/BiasAdd_3:output:0$while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????du
while/lstm_cell_6/Sigmoid_2Sigmoidwhile/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????do
while/lstm_cell_6/Tanh_1Tanhwhile/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
while/lstm_cell_6/mul_10Mulwhile/lstm_cell_6/Sigmoid_2:y:0while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????de
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
while/SelectV2SelectV2while/Tile:output:0while/lstm_cell_6/mul_10:z:0while_placeholder_2*
T0*'
_output_shapes
:?????????dg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????g
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:??????????
while/SelectV2_1SelectV2while/Tile_1:output:0while/lstm_cell_6/mul_10:z:0while_placeholder_3*
T0*'
_output_shapes
:?????????d?
while/SelectV2_2SelectV2while/Tile_2:output:0while/lstm_cell_6/add_3:z:0while_placeholder_4*
T0*'
_output_shapes
:?????????dr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: t
while/Identity_4Identitywhile/SelectV2:output:0^while/NoOp*
T0*'
_output_shapes
:?????????dv
while/Identity_5Identitywhile/SelectV2_1:output:0^while/NoOp*
T0*'
_output_shapes
:?????????dv
while/Identity_6Identitywhile/SelectV2_2:output:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp!^while/lstm_cell_6/ReadVariableOp#^while/lstm_cell_6/ReadVariableOp_1#^while/lstm_cell_6/ReadVariableOp_2#^while/lstm_cell_6/ReadVariableOp_3'^while/lstm_cell_6/split/ReadVariableOp)^while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"X
)while_lstm_cell_6_readvariableop_resource+while_lstm_cell_6_readvariableop_resource_0"h
1while_lstm_cell_6_split_1_readvariableop_resource3while_lstm_cell_6_split_1_readvariableop_resource_0"d
/while_lstm_cell_6_split_readvariableop_resource1while_lstm_cell_6_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : 2D
 while/lstm_cell_6/ReadVariableOp while/lstm_cell_6/ReadVariableOp2H
"while/lstm_cell_6/ReadVariableOp_1"while/lstm_cell_6/ReadVariableOp_12H
"while/lstm_cell_6/ReadVariableOp_2"while/lstm_cell_6/ReadVariableOp_22H
"while/lstm_cell_6/ReadVariableOp_3"while/lstm_cell_6/ReadVariableOp_32P
&while/lstm_cell_6/split/ReadVariableOp&while/lstm_cell_6/split/ReadVariableOp2T
(while/lstm_cell_6/split_1/ReadVariableOp(while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?

?
'__inference_restore_from_tensors_610178M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: @
<mutablehashtable_table_restore_lookuptableimportv2_restorev2B
>mutablehashtable_table_restore_lookuptableimportv2_restorev2_1	
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable<mutablehashtable_table_restore_lookuptableimportv2_restorev2>mutablehashtable_table_restore_lookuptableimportv2_restorev2_1*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes

: ::2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:) %
#
_class
loc:@MutableHashTable:C?
#
_class
loc:@MutableHashTable

_output_shapes
::C?
#
_class
loc:@MutableHashTable

_output_shapes
:
?t
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607049

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	%
embedding_1_607025:	?@ 
lstm_2_607030:	@?
lstm_2_607032:	? 
lstm_2_607034:	d?!
dense_2_607037:	d?
dense_2_607039:	?!
dense_3_607043:	?
dense_3_607045:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
gtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ptext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountjtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????????????
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1_607025*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_606111X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
embedding_1/NotEqualNotEqual?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*0
_output_shapes
:???????????????????
lstm_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0lstm_2_607030lstm_2_607032lstm_2_607034*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_606925?
dense_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_2_607037dense_2_607039*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_606408?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_606496?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_607043dense_3_607045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_606432w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_dense_2_layer_call_fn_609585

inputs
unknown:	d?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_606408p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?K
?
__inference__traced_save_610102
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop8
4savev2_lstm_2_lstm_cell_6_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_6_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_6_kernel_m_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_6_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_6_bias_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_6_kernel_v_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_6_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_6_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop4savev2_lstm_2_lstm_cell_6_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_6_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop;savev2_adam_lstm_2_lstm_cell_6_kernel_m_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_6_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_2_lstm_cell_6_bias_m_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop;savev2_adam_lstm_2_lstm_cell_6_kernel_v_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_6_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_2_lstm_cell_6_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?@:	d?:?:	?::	@?:	d?:?: : : : : ::: : : : :	?@:	d?:?:	?::	@?:	d?:?:	?@:	d?:?:	?::	@?:	d?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?@:%!

_output_shapes
:	d?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	@?:%!

_output_shapes
:	d?:!

_output_shapes	
:?:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?@:%!

_output_shapes
:	d?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	@?:%!

_output_shapes
:	d?:!

_output_shapes	
:?:%!

_output_shapes
:	?@:%!

_output_shapes
:	d?:!

_output_shapes	
:?:%!

_output_shapes
:	?:  

_output_shapes
::%!!

_output_shapes
:	@?:%"!

_output_shapes
:	d?:!#

_output_shapes	
:?:$

_output_shapes
: 
?
?
__inference_save_fn_609957
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_609623

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_609643

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_605730

inputs%
lstm_cell_6_605646:	@?!
lstm_cell_6_605648:	?%
lstm_cell_6_605650:	d?
identity??#lstm_cell_6/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_mask?
#lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_6_605646lstm_cell_6_605648lstm_cell_6_605650*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_605645n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_6_605646lstm_cell_6_605648lstm_cell_6_605650*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_605660*
condR
while_cond_605659*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????dt
NoOpNoOp$^lstm_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 2J
#lstm_cell_6/StatefulPartitionedCall#lstm_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
??
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_606389

inputs
mask
<
)lstm_cell_6_split_readvariableop_resource:	@?:
+lstm_cell_6_split_1_readvariableop_resource:	?6
#lstm_cell_6_readvariableop_resource:	d?
identity??lstm_cell_6/ReadVariableOp?lstm_cell_6/ReadVariableOp_1?lstm_cell_6/ReadVariableOp_2?lstm_cell_6/ReadVariableOp_3? lstm_cell_6/split/ReadVariableOp?"lstm_cell_6/split_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????v

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskc
lstm_cell_6/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_likeFill$lstm_cell_6/ones_like/Shape:output:0$lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@[
lstm_cell_6/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_like_1Fill&lstm_cell_6/ones_like_1/Shape:output:0&lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/mulMulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_1Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_2Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_3Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_6/split/ReadVariableOpReadVariableOp)lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0(lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split
lstm_cell_6/MatMulMatMullstm_cell_6/mul:z:0lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_1MatMullstm_cell_6/mul_1:z:0lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_2MatMullstm_cell_6/mul_2:z:0lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_3MatMullstm_cell_6/mul_3:z:0lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????d_
lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_6/split_1/ReadVariableOpReadVariableOp+lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_6/split_1Split&lstm_cell_6/split_1/split_dim:output:0*lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
lstm_cell_6/BiasAddBiasAddlstm_cell_6/MatMul:product:0lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_1BiasAddlstm_cell_6/MatMul_1:product:0lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_2BiasAddlstm_cell_6/MatMul_2:product:0lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_3BiasAddlstm_cell_6/MatMul_3:product:0lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_4Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_5Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_6Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_7Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d
lstm_cell_6/ReadVariableOpReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0p
lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   r
!lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_sliceStridedSlice"lstm_cell_6/ReadVariableOp:value:0(lstm_cell_6/strided_slice/stack:output:0*lstm_cell_6/strided_slice/stack_1:output:0*lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_4MatMullstm_cell_6/mul_4:z:0"lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/addAddV2lstm_cell_6/BiasAdd:output:0lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????de
lstm_cell_6/SigmoidSigmoidlstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_1ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   t
#lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_1StridedSlice$lstm_cell_6/ReadVariableOp_1:value:0*lstm_cell_6/strided_slice_1/stack:output:0,lstm_cell_6/strided_slice_1/stack_1:output:0,lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_5MatMullstm_cell_6/mul_5:z:0$lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_1AddV2lstm_cell_6/BiasAdd_1:output:0lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????dw
lstm_cell_6/mul_8Mullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_2ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_2StridedSlice$lstm_cell_6/ReadVariableOp_2:value:0*lstm_cell_6/strided_slice_2/stack:output:0,lstm_cell_6/strided_slice_2/stack_1:output:0,lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_6MatMullstm_cell_6/mul_6:z:0$lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_2AddV2lstm_cell_6/BiasAdd_2:output:0lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????da
lstm_cell_6/TanhTanhlstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????dy
lstm_cell_6/mul_9Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????dz
lstm_cell_6/add_3AddV2lstm_cell_6/mul_8:z:0lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_3ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_3StridedSlice$lstm_cell_6/ReadVariableOp_3:value:0*lstm_cell_6/strided_slice_3/stack:output:0,lstm_cell_6/strided_slice_3/stack_1:output:0,lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_7MatMullstm_cell_6/mul_7:z:0$lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_4AddV2lstm_cell_6/BiasAdd_3:output:0lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????dc
lstm_cell_6/Tanh_1Tanhlstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d~
lstm_cell_6/mul_10Mullstm_cell_6/Sigmoid_2:y:0lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:???a

zeros_like	ZerosLikelstm_cell_6/mul_10:z:0*
T0*'
_output_shapes
:?????????dc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0)lstm_cell_6_split_readvariableop_resource+lstm_cell_6_split_1_readvariableop_resource#lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *
bodyR
while_body_606236*
condR
while_cond_606235*`
output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^lstm_cell_6/ReadVariableOp^lstm_cell_6/ReadVariableOp_1^lstm_cell_6/ReadVariableOp_2^lstm_cell_6/ReadVariableOp_3!^lstm_cell_6/split/ReadVariableOp#^lstm_cell_6/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????????????@:??????????????????: : : 28
lstm_cell_6/ReadVariableOplstm_cell_6/ReadVariableOp2<
lstm_cell_6/ReadVariableOp_1lstm_cell_6/ReadVariableOp_12<
lstm_cell_6/ReadVariableOp_2lstm_cell_6/ReadVariableOp_22<
lstm_cell_6/ReadVariableOp_3lstm_cell_6/ReadVariableOp_32D
 lstm_cell_6/split/ReadVariableOp lstm_cell_6/split/ReadVariableOp2H
"lstm_cell_6/split_1/ReadVariableOp"lstm_cell_6/split_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
?#
?
while_body_605967
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_6_605991_0:	@?)
while_lstm_cell_6_605993_0:	?-
while_lstm_cell_6_605995_0:	d?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_6_605991:	@?'
while_lstm_cell_6_605993:	?+
while_lstm_cell_6_605995:	d???)while/lstm_cell_6/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
)while/lstm_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_6_605991_0while_lstm_cell_6_605993_0while_lstm_cell_6_605995_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_605907r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_6/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_4Identity2while/lstm_cell_6/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?
while/Identity_5Identity2while/lstm_cell_6/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:?????????dx

while/NoOpNoOp*^while/lstm_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_6_605991while_lstm_cell_6_605991_0"6
while_lstm_cell_6_605993while_lstm_cell_6_605993_0"6
while_lstm_cell_6_605995while_lstm_cell_6_605995_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :?????????d:?????????d: : : : : 2V
)while/lstm_cell_6/StatefulPartitionedCall)while/lstm_cell_6/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
-
__inference__destroyer_609923
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?s
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607181
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	%
embedding_1_607157:	?@ 
lstm_2_607162:	@?
lstm_2_607164:	? 
lstm_2_607166:	d?!
dense_2_607169:	d?
dense_2_607171:	?!
dense_3_607175:	?
dense_3_607177:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
gtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ptext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountjtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????????????
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1_607157*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_606111X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
embedding_1/NotEqualNotEqual?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*0
_output_shapes
:???????????????????
lstm_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0lstm_2_607162lstm_2_607164lstm_2_607166*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_606389?
dense_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_2_607169dense_2_607171*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_606408?
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_606419?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_607175dense_3_607177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_606432w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_lstm_cell_6_layer_call_fn_609677

inputs
states_0
states_1
unknown:	@?
	unknown_0:	?
	unknown_1:	d?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_605907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????dq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:?????????d:?????????d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/1
?	
?
while_cond_606707
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_14
0while_while_cond_606707___redundant_placeholder04
0while_while_cond_606707___redundant_placeholder14
0while_while_cond_606707___redundant_placeholder24
0while_while_cond_606707___redundant_placeholder34
0while_while_cond_606707___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :?????????d:?????????d:?????????d: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
Ѝ
?

H__inference_sequential_1_layer_call_and_return_conditional_losses_607744

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	6
#embedding_1_embedding_lookup_607453:	?@C
0lstm_2_lstm_cell_6_split_readvariableop_resource:	@?A
2lstm_2_lstm_cell_6_split_1_readvariableop_resource:	?=
*lstm_2_lstm_cell_6_readvariableop_resource:	d?9
&dense_2_matmul_readvariableop_resource:	d?6
'dense_2_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookup?!lstm_2/lstm_cell_6/ReadVariableOp?#lstm_2/lstm_cell_6/ReadVariableOp_1?#lstm_2/lstm_cell_6/ReadVariableOp_2?#lstm_2/lstm_cell_6/ReadVariableOp_3?'lstm_2/lstm_cell_6/split/ReadVariableOp?)lstm_2/lstm_cell_6/split_1/ReadVariableOp?lstm_2/while?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
gtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ptext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountjtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????????????
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_607453?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*6
_class,
*(loc:@embedding_1/embedding_lookup/607453*4
_output_shapes"
 :??????????????????@*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/607453*4
_output_shapes"
 :??????????????????@?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
embedding_1/NotEqualNotEqual?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*0
_output_shapes
:??????????????????l
lstm_2/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:d
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dY
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dj
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0lstm_2/transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@R
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:f
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
lstm_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/ExpandDims
ExpandDimsembedding_1/NotEqual:z:0lstm_2/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :??????????????????l
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose_1	Transposelstm_2/ExpandDims:output:0 lstm_2/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :??????????????????m
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???f
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskq
"lstm_2/lstm_cell_6/ones_like/ShapeShapelstm_2/strided_slice_2:output:0*
T0*
_output_shapes
:g
"lstm_2/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_2/lstm_cell_6/ones_likeFill+lstm_2/lstm_cell_6/ones_like/Shape:output:0+lstm_2/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@i
$lstm_2/lstm_cell_6/ones_like_1/ShapeShapelstm_2/zeros:output:0*
T0*
_output_shapes
:i
$lstm_2/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_2/lstm_cell_6/ones_like_1Fill-lstm_2/lstm_cell_6/ones_like_1/Shape:output:0-lstm_2/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mulMullstm_2/strided_slice_2:output:0%lstm_2/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_2/lstm_cell_6/mul_1Mullstm_2/strided_slice_2:output:0%lstm_2/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_2/lstm_cell_6/mul_2Mullstm_2/strided_slice_2:output:0%lstm_2/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_2/lstm_cell_6/mul_3Mullstm_2/strided_slice_2:output:0%lstm_2/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@d
"lstm_2/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
'lstm_2/lstm_cell_6/split/ReadVariableOpReadVariableOp0lstm_2_lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_2/lstm_cell_6/splitSplit+lstm_2/lstm_cell_6/split/split_dim:output:0/lstm_2/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
lstm_2/lstm_cell_6/MatMulMatMullstm_2/lstm_cell_6/mul:z:0!lstm_2/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/MatMul_1MatMullstm_2/lstm_cell_6/mul_1:z:0!lstm_2/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/MatMul_2MatMullstm_2/lstm_cell_6/mul_2:z:0!lstm_2/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/MatMul_3MatMullstm_2/lstm_cell_6/mul_3:z:0!lstm_2/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????df
$lstm_2/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
)lstm_2/lstm_cell_6/split_1/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_2/lstm_cell_6/split_1Split-lstm_2/lstm_cell_6/split_1/split_dim:output:01lstm_2/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
lstm_2/lstm_cell_6/BiasAddBiasAdd#lstm_2/lstm_cell_6/MatMul:product:0#lstm_2/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/BiasAdd_1BiasAdd%lstm_2/lstm_cell_6/MatMul_1:product:0#lstm_2/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/BiasAdd_2BiasAdd%lstm_2/lstm_cell_6/MatMul_2:product:0#lstm_2/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/BiasAdd_3BiasAdd%lstm_2/lstm_cell_6/MatMul_3:product:0#lstm_2/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_4Mullstm_2/zeros:output:0'lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_5Mullstm_2/zeros:output:0'lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_6Mullstm_2/zeros:output:0'lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_7Mullstm_2/zeros:output:0'lstm_2/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
!lstm_2/lstm_cell_6/ReadVariableOpReadVariableOp*lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0w
&lstm_2/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_2/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   y
(lstm_2/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
 lstm_2/lstm_cell_6/strided_sliceStridedSlice)lstm_2/lstm_cell_6/ReadVariableOp:value:0/lstm_2/lstm_cell_6/strided_slice/stack:output:01lstm_2/lstm_cell_6/strided_slice/stack_1:output:01lstm_2/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_2/lstm_cell_6/MatMul_4MatMullstm_2/lstm_cell_6/mul_4:z:0)lstm_2/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/addAddV2#lstm_2/lstm_cell_6/BiasAdd:output:0%lstm_2/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????ds
lstm_2/lstm_cell_6/SigmoidSigmoidlstm_2/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
#lstm_2/lstm_cell_6/ReadVariableOp_1ReadVariableOp*lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0y
(lstm_2/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   {
*lstm_2/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   {
*lstm_2/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_6/strided_slice_1StridedSlice+lstm_2/lstm_cell_6/ReadVariableOp_1:value:01lstm_2/lstm_cell_6/strided_slice_1/stack:output:03lstm_2/lstm_cell_6/strided_slice_1/stack_1:output:03lstm_2/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_2/lstm_cell_6/MatMul_5MatMullstm_2/lstm_cell_6/mul_5:z:0+lstm_2/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/add_1AddV2%lstm_2/lstm_cell_6/BiasAdd_1:output:0%lstm_2/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????dw
lstm_2/lstm_cell_6/Sigmoid_1Sigmoidlstm_2/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_8Mul lstm_2/lstm_cell_6/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
#lstm_2/lstm_cell_6/ReadVariableOp_2ReadVariableOp*lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0y
(lstm_2/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   {
*lstm_2/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  {
*lstm_2/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_6/strided_slice_2StridedSlice+lstm_2/lstm_cell_6/ReadVariableOp_2:value:01lstm_2/lstm_cell_6/strided_slice_2/stack:output:03lstm_2/lstm_cell_6/strided_slice_2/stack_1:output:03lstm_2/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_2/lstm_cell_6/MatMul_6MatMullstm_2/lstm_cell_6/mul_6:z:0+lstm_2/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/add_2AddV2%lstm_2/lstm_cell_6/BiasAdd_2:output:0%lstm_2/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????do
lstm_2/lstm_cell_6/TanhTanhlstm_2/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_9Mullstm_2/lstm_cell_6/Sigmoid:y:0lstm_2/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/add_3AddV2lstm_2/lstm_cell_6/mul_8:z:0lstm_2/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
#lstm_2/lstm_cell_6/ReadVariableOp_3ReadVariableOp*lstm_2_lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0y
(lstm_2/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  {
*lstm_2/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_2/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
"lstm_2/lstm_cell_6/strided_slice_3StridedSlice+lstm_2/lstm_cell_6/ReadVariableOp_3:value:01lstm_2/lstm_cell_6/strided_slice_3/stack:output:03lstm_2/lstm_cell_6/strided_slice_3/stack_1:output:03lstm_2/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_2/lstm_cell_6/MatMul_7MatMullstm_2/lstm_cell_6/mul_7:z:0+lstm_2/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/add_4AddV2%lstm_2/lstm_cell_6/BiasAdd_3:output:0%lstm_2/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????dw
lstm_2/lstm_cell_6/Sigmoid_2Sigmoidlstm_2/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????dq
lstm_2/lstm_cell_6/Tanh_1Tanhlstm_2/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/lstm_cell_6/mul_10Mul lstm_2/lstm_cell_6/Sigmoid_2:y:0lstm_2/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????du
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   e
#lstm_2/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0,lstm_2/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???M
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : o
$lstm_2/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
lstm_2/TensorArrayV2_2TensorListReserve-lstm_2/TensorArrayV2_2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:????
>lstm_2/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
0lstm_2/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorlstm_2/transpose_1:y:0Glstm_2/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:???o
lstm_2/zeros_like	ZerosLikelstm_2/lstm_cell_6/mul_10:z:0*
T0*'
_output_shapes
:?????????dj
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????[
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros_like:y:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0@lstm_2/TensorArrayUnstack_1/TensorListFromTensor:output_handle:00lstm_2_lstm_cell_6_split_readvariableop_resource2lstm_2_lstm_cell_6_split_1_readvariableop_resource*lstm_2_lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *%
_read_only_resource_inputs

*
_stateful_parallelism( *$
bodyR
lstm_2_while_body_607576*$
condR
lstm_2_while_cond_607575*`
output_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : *
parallel_iterations ?
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementso
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maskl
lstm_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
lstm_2/transpose_2	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:?????????db
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
dense_2/MatMulMatMullstm_2/strided_slice_3:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????m
dropout_1/IdentityIdentitydense_2/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMuldropout_1/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup"^lstm_2/lstm_cell_6/ReadVariableOp$^lstm_2/lstm_cell_6/ReadVariableOp_1$^lstm_2/lstm_cell_6/ReadVariableOp_2$^lstm_2/lstm_cell_6/ReadVariableOp_3(^lstm_2/lstm_cell_6/split/ReadVariableOp*^lstm_2/lstm_cell_6/split_1/ReadVariableOp^lstm_2/while?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2F
!lstm_2/lstm_cell_6/ReadVariableOp!lstm_2/lstm_cell_6/ReadVariableOp2J
#lstm_2/lstm_cell_6/ReadVariableOp_1#lstm_2/lstm_cell_6/ReadVariableOp_12J
#lstm_2/lstm_cell_6/ReadVariableOp_2#lstm_2/lstm_cell_6/ReadVariableOp_22J
#lstm_2/lstm_cell_6/ReadVariableOp_3#lstm_2/lstm_cell_6/ReadVariableOp_32R
'lstm_2/lstm_cell_6/split/ReadVariableOp'lstm_2/lstm_cell_6/split/ReadVariableOp2V
)lstm_2/lstm_cell_6/split_1/ReadVariableOp)lstm_2/lstm_cell_6/split_1/ReadVariableOp2
lstm_2/whilelstm_2/while2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
G__inference_embedding_1_layer_call_and_return_conditional_losses_608238

inputs	*
embedding_lookup_608232:	?@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_608232inputs*
Tindices0	**
_class 
loc:@embedding_lookup/608232*4
_output_shapes"
 :??????????????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/608232*4
_output_shapes"
 :??????????????????@?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :??????????????????@?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????????????: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
??
?
lstm_2_while_body_607983*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3
lstm_2_while_placeholder_4)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0i
elstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0K
8lstm_2_while_lstm_cell_6_split_readvariableop_resource_0:	@?I
:lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0:	?E
2lstm_2_while_lstm_cell_6_readvariableop_resource_0:	d?
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5
lstm_2_while_identity_6'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorg
clstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensorI
6lstm_2_while_lstm_cell_6_split_readvariableop_resource:	@?G
8lstm_2_while_lstm_cell_6_split_1_readvariableop_resource:	?C
0lstm_2_while_lstm_cell_6_readvariableop_resource:	d???'lstm_2/while/lstm_cell_6/ReadVariableOp?)lstm_2/while/lstm_cell_6/ReadVariableOp_1?)lstm_2/while/lstm_cell_6/ReadVariableOp_2?)lstm_2/while/lstm_cell_6/ReadVariableOp_3?-lstm_2/while/lstm_cell_6/split/ReadVariableOp?/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp?
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
@lstm_2/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
2lstm_2/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0lstm_2_while_placeholderIlstm_2/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
(lstm_2/while/lstm_cell_6/ones_like/ShapeShape7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(lstm_2/while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm_2/while/lstm_cell_6/ones_likeFill1lstm_2/while/lstm_cell_6/ones_like/Shape:output:01lstm_2/while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@k
&lstm_2/while/lstm_cell_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm_2/while/lstm_cell_6/dropout/MulMul+lstm_2/while/lstm_cell_6/ones_like:output:0/lstm_2/while/lstm_cell_6/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@?
&lstm_2/while/lstm_cell_6/dropout/ShapeShape+lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
=lstm_2/while/lstm_cell_6/dropout/random_uniform/RandomUniformRandomUniform/lstm_2/while/lstm_cell_6/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0t
/lstm_2/while/lstm_cell_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
-lstm_2/while/lstm_cell_6/dropout/GreaterEqualGreaterEqualFlstm_2/while/lstm_cell_6/dropout/random_uniform/RandomUniform:output:08lstm_2/while/lstm_cell_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
%lstm_2/while/lstm_cell_6/dropout/CastCast1lstm_2/while/lstm_cell_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
&lstm_2/while/lstm_cell_6/dropout/Mul_1Mul(lstm_2/while/lstm_cell_6/dropout/Mul:z:0)lstm_2/while/lstm_cell_6/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@m
(lstm_2/while/lstm_cell_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_2/while/lstm_cell_6/dropout_1/MulMul+lstm_2/while/lstm_cell_6/ones_like:output:01lstm_2/while/lstm_cell_6/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@?
(lstm_2/while/lstm_cell_6/dropout_1/ShapeShape+lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_2/while/lstm_cell_6/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_6/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0v
1lstm_2/while/lstm_cell_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_2/while/lstm_cell_6/dropout_1/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_6/dropout_1/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_6/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
'lstm_2/while/lstm_cell_6/dropout_1/CastCast3lstm_2/while/lstm_cell_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
(lstm_2/while/lstm_cell_6/dropout_1/Mul_1Mul*lstm_2/while/lstm_cell_6/dropout_1/Mul:z:0+lstm_2/while/lstm_cell_6/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@m
(lstm_2/while/lstm_cell_6/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_2/while/lstm_cell_6/dropout_2/MulMul+lstm_2/while/lstm_cell_6/ones_like:output:01lstm_2/while/lstm_cell_6/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@?
(lstm_2/while/lstm_cell_6/dropout_2/ShapeShape+lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_2/while/lstm_cell_6/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_6/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0v
1lstm_2/while/lstm_cell_6/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_2/while/lstm_cell_6/dropout_2/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_6/dropout_2/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_6/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
'lstm_2/while/lstm_cell_6/dropout_2/CastCast3lstm_2/while/lstm_cell_6/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
(lstm_2/while/lstm_cell_6/dropout_2/Mul_1Mul*lstm_2/while/lstm_cell_6/dropout_2/Mul:z:0+lstm_2/while/lstm_cell_6/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@m
(lstm_2/while/lstm_cell_6/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_2/while/lstm_cell_6/dropout_3/MulMul+lstm_2/while/lstm_cell_6/ones_like:output:01lstm_2/while/lstm_cell_6/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@?
(lstm_2/while/lstm_cell_6/dropout_3/ShapeShape+lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
?lstm_2/while/lstm_cell_6/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_6/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0v
1lstm_2/while/lstm_cell_6/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_2/while/lstm_cell_6/dropout_3/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_6/dropout_3/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_6/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
'lstm_2/while/lstm_cell_6/dropout_3/CastCast3lstm_2/while/lstm_cell_6/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
(lstm_2/while/lstm_cell_6/dropout_3/Mul_1Mul*lstm_2/while/lstm_cell_6/dropout_3/Mul:z:0+lstm_2/while/lstm_cell_6/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@t
*lstm_2/while/lstm_cell_6/ones_like_1/ShapeShapelstm_2_while_placeholder_3*
T0*
_output_shapes
:o
*lstm_2/while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm_2/while/lstm_cell_6/ones_like_1Fill3lstm_2/while/lstm_cell_6/ones_like_1/Shape:output:03lstm_2/while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????dm
(lstm_2/while/lstm_cell_6/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_2/while/lstm_cell_6/dropout_4/MulMul-lstm_2/while/lstm_cell_6/ones_like_1:output:01lstm_2/while/lstm_cell_6/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????d?
(lstm_2/while/lstm_cell_6/dropout_4/ShapeShape-lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
?lstm_2/while/lstm_cell_6/dropout_4/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_6/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0v
1lstm_2/while/lstm_cell_6/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_2/while/lstm_cell_6/dropout_4/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_6/dropout_4/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_6/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
'lstm_2/while/lstm_cell_6/dropout_4/CastCast3lstm_2/while/lstm_cell_6/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
(lstm_2/while/lstm_cell_6/dropout_4/Mul_1Mul*lstm_2/while/lstm_cell_6/dropout_4/Mul:z:0+lstm_2/while/lstm_cell_6/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????dm
(lstm_2/while/lstm_cell_6/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_2/while/lstm_cell_6/dropout_5/MulMul-lstm_2/while/lstm_cell_6/ones_like_1:output:01lstm_2/while/lstm_cell_6/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????d?
(lstm_2/while/lstm_cell_6/dropout_5/ShapeShape-lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
?lstm_2/while/lstm_cell_6/dropout_5/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_6/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0v
1lstm_2/while/lstm_cell_6/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_2/while/lstm_cell_6/dropout_5/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_6/dropout_5/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_6/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
'lstm_2/while/lstm_cell_6/dropout_5/CastCast3lstm_2/while/lstm_cell_6/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
(lstm_2/while/lstm_cell_6/dropout_5/Mul_1Mul*lstm_2/while/lstm_cell_6/dropout_5/Mul:z:0+lstm_2/while/lstm_cell_6/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????dm
(lstm_2/while/lstm_cell_6/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_2/while/lstm_cell_6/dropout_6/MulMul-lstm_2/while/lstm_cell_6/ones_like_1:output:01lstm_2/while/lstm_cell_6/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????d?
(lstm_2/while/lstm_cell_6/dropout_6/ShapeShape-lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
?lstm_2/while/lstm_cell_6/dropout_6/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_6/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0v
1lstm_2/while/lstm_cell_6/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_2/while/lstm_cell_6/dropout_6/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_6/dropout_6/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_6/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
'lstm_2/while/lstm_cell_6/dropout_6/CastCast3lstm_2/while/lstm_cell_6/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
(lstm_2/while/lstm_cell_6/dropout_6/Mul_1Mul*lstm_2/while/lstm_cell_6/dropout_6/Mul:z:0+lstm_2/while/lstm_cell_6/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????dm
(lstm_2/while/lstm_cell_6/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&lstm_2/while/lstm_cell_6/dropout_7/MulMul-lstm_2/while/lstm_cell_6/ones_like_1:output:01lstm_2/while/lstm_cell_6/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????d?
(lstm_2/while/lstm_cell_6/dropout_7/ShapeShape-lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
?lstm_2/while/lstm_cell_6/dropout_7/random_uniform/RandomUniformRandomUniform1lstm_2/while/lstm_cell_6/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0v
1lstm_2/while/lstm_cell_6/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
/lstm_2/while/lstm_cell_6/dropout_7/GreaterEqualGreaterEqualHlstm_2/while/lstm_cell_6/dropout_7/random_uniform/RandomUniform:output:0:lstm_2/while/lstm_cell_6/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
'lstm_2/while/lstm_cell_6/dropout_7/CastCast3lstm_2/while/lstm_cell_6/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
(lstm_2/while/lstm_cell_6/dropout_7/Mul_1Mul*lstm_2/while/lstm_cell_6/dropout_7/Mul:z:0+lstm_2/while/lstm_cell_6/dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mulMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0*lstm_2/while/lstm_cell_6/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_2/while/lstm_cell_6/mul_1Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_2/while/lstm_cell_6/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_2/while/lstm_cell_6/mul_2Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_2/while/lstm_cell_6/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_2/while/lstm_cell_6/mul_3Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_2/while/lstm_cell_6/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@j
(lstm_2/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_2/while/lstm_cell_6/split/ReadVariableOpReadVariableOp8lstm_2_while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
lstm_2/while/lstm_cell_6/splitSplit1lstm_2/while/lstm_cell_6/split/split_dim:output:05lstm_2/while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
lstm_2/while/lstm_cell_6/MatMulMatMul lstm_2/while/lstm_cell_6/mul:z:0'lstm_2/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
!lstm_2/while/lstm_cell_6/MatMul_1MatMul"lstm_2/while/lstm_cell_6/mul_1:z:0'lstm_2/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
!lstm_2/while/lstm_cell_6/MatMul_2MatMul"lstm_2/while/lstm_cell_6/mul_2:z:0'lstm_2/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
!lstm_2/while/lstm_cell_6/MatMul_3MatMul"lstm_2/while/lstm_cell_6/mul_3:z:0'lstm_2/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????dl
*lstm_2/while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_2/while/lstm_cell_6/split_1/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
 lstm_2/while/lstm_cell_6/split_1Split3lstm_2/while/lstm_cell_6/split_1/split_dim:output:07lstm_2/while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
 lstm_2/while/lstm_cell_6/BiasAddBiasAdd)lstm_2/while/lstm_cell_6/MatMul:product:0)lstm_2/while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/BiasAdd_1BiasAdd+lstm_2/while/lstm_cell_6/MatMul_1:product:0)lstm_2/while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/BiasAdd_2BiasAdd+lstm_2/while/lstm_cell_6/MatMul_2:product:0)lstm_2/while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/BiasAdd_3BiasAdd+lstm_2/while/lstm_cell_6/MatMul_3:product:0)lstm_2/while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_4Mullstm_2_while_placeholder_3,lstm_2/while/lstm_cell_6/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_5Mullstm_2_while_placeholder_3,lstm_2/while/lstm_cell_6/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_6Mullstm_2_while_placeholder_3,lstm_2/while/lstm_cell_6/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_7Mullstm_2_while_placeholder_3,lstm_2/while/lstm_cell_6/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????d?
'lstm_2/while/lstm_cell_6/ReadVariableOpReadVariableOp2lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0}
,lstm_2/while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_2/while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   
.lstm_2/while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_2/while/lstm_cell_6/strided_sliceStridedSlice/lstm_2/while/lstm_cell_6/ReadVariableOp:value:05lstm_2/while/lstm_cell_6/strided_slice/stack:output:07lstm_2/while/lstm_cell_6/strided_slice/stack_1:output:07lstm_2/while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_6/MatMul_4MatMul"lstm_2/while/lstm_cell_6/mul_4:z:0/lstm_2/while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/addAddV2)lstm_2/while/lstm_cell_6/BiasAdd:output:0+lstm_2/while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????d
 lstm_2/while/lstm_cell_6/SigmoidSigmoid lstm_2/while/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
)lstm_2/while/lstm_cell_6/ReadVariableOp_1ReadVariableOp2lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0
.lstm_2/while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   ?
0lstm_2/while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   ?
0lstm_2/while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_6/strided_slice_1StridedSlice1lstm_2/while/lstm_cell_6/ReadVariableOp_1:value:07lstm_2/while/lstm_cell_6/strided_slice_1/stack:output:09lstm_2/while/lstm_cell_6/strided_slice_1/stack_1:output:09lstm_2/while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_6/MatMul_5MatMul"lstm_2/while/lstm_cell_6/mul_5:z:01lstm_2/while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/add_1AddV2+lstm_2/while/lstm_cell_6/BiasAdd_1:output:0+lstm_2/while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/Sigmoid_1Sigmoid"lstm_2/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_8Mul&lstm_2/while/lstm_cell_6/Sigmoid_1:y:0lstm_2_while_placeholder_4*
T0*'
_output_shapes
:?????????d?
)lstm_2/while/lstm_cell_6/ReadVariableOp_2ReadVariableOp2lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0
.lstm_2/while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
0lstm_2/while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  ?
0lstm_2/while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_6/strided_slice_2StridedSlice1lstm_2/while/lstm_cell_6/ReadVariableOp_2:value:07lstm_2/while/lstm_cell_6/strided_slice_2/stack:output:09lstm_2/while/lstm_cell_6/strided_slice_2/stack_1:output:09lstm_2/while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_6/MatMul_6MatMul"lstm_2/while/lstm_cell_6/mul_6:z:01lstm_2/while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/add_2AddV2+lstm_2/while/lstm_cell_6/BiasAdd_2:output:0+lstm_2/while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????d{
lstm_2/while/lstm_cell_6/TanhTanh"lstm_2/while/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_9Mul$lstm_2/while/lstm_cell_6/Sigmoid:y:0!lstm_2/while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/add_3AddV2"lstm_2/while/lstm_cell_6/mul_8:z:0"lstm_2/while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
)lstm_2/while/lstm_cell_6/ReadVariableOp_3ReadVariableOp2lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0
.lstm_2/while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  ?
0lstm_2/while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_2/while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_6/strided_slice_3StridedSlice1lstm_2/while/lstm_cell_6/ReadVariableOp_3:value:07lstm_2/while/lstm_cell_6/strided_slice_3/stack:output:09lstm_2/while/lstm_cell_6/strided_slice_3/stack_1:output:09lstm_2/while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_6/MatMul_7MatMul"lstm_2/while/lstm_cell_6/mul_7:z:01lstm_2/while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/add_4AddV2+lstm_2/while/lstm_cell_6/BiasAdd_3:output:0+lstm_2/while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/Sigmoid_2Sigmoid"lstm_2/while/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????d}
lstm_2/while/lstm_cell_6/Tanh_1Tanh"lstm_2/while/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_10Mul&lstm_2/while/lstm_cell_6/Sigmoid_2:y:0#lstm_2/while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dl
lstm_2/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_2/while/TileTile9lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_2/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
lstm_2/while/SelectV2SelectV2lstm_2/while/Tile:output:0#lstm_2/while/lstm_cell_6/mul_10:z:0lstm_2_while_placeholder_2*
T0*'
_output_shapes
:?????????dn
lstm_2/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_2/while/Tile_1Tile9lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_2/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????n
lstm_2/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_2/while/Tile_2Tile9lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_2/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:??????????
lstm_2/while/SelectV2_1SelectV2lstm_2/while/Tile_1:output:0#lstm_2/while/lstm_cell_6/mul_10:z:0lstm_2_while_placeholder_3*
T0*'
_output_shapes
:?????????d?
lstm_2/while/SelectV2_2SelectV2lstm_2/while/Tile_2:output:0"lstm_2/while/lstm_cell_6/add_3:z:0lstm_2_while_placeholder_4*
T0*'
_output_shapes
:?????????dy
7lstm_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1@lstm_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0lstm_2/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???T
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: n
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_4Identitylstm_2/while/SelectV2:output:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
lstm_2/while/Identity_5Identity lstm_2/while/SelectV2_1:output:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
lstm_2/while/Identity_6Identity lstm_2/while/SelectV2_2:output:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
lstm_2/while/NoOpNoOp(^lstm_2/while/lstm_cell_6/ReadVariableOp*^lstm_2/while/lstm_cell_6/ReadVariableOp_1*^lstm_2/while/lstm_cell_6/ReadVariableOp_2*^lstm_2/while/lstm_cell_6/ReadVariableOp_3.^lstm_2/while/lstm_cell_6/split/ReadVariableOp0^lstm_2/while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0";
lstm_2_while_identity_6 lstm_2/while/Identity_6:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"f
0lstm_2_while_lstm_cell_6_readvariableop_resource2lstm_2_while_lstm_cell_6_readvariableop_resource_0"v
8lstm_2_while_lstm_cell_6_split_1_readvariableop_resource:lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0"r
6lstm_2_while_lstm_cell_6_split_readvariableop_resource8lstm_2_while_lstm_cell_6_split_readvariableop_resource_0"?
clstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensorelstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0"?
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : 2R
'lstm_2/while/lstm_cell_6/ReadVariableOp'lstm_2/while/lstm_cell_6/ReadVariableOp2V
)lstm_2/while/lstm_cell_6/ReadVariableOp_1)lstm_2/while/lstm_cell_6/ReadVariableOp_12V
)lstm_2/while/lstm_cell_6/ReadVariableOp_2)lstm_2/while/lstm_cell_6/ReadVariableOp_22V
)lstm_2/while/lstm_cell_6/ReadVariableOp_3)lstm_2/while/lstm_cell_6/ReadVariableOp_32^
-lstm_2/while/lstm_cell_6/split/ReadVariableOp-lstm_2/while/lstm_cell_6/split/ReadVariableOp2b
/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
/
__inference__initializer_609933
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_6099187
3key_value_init8291_lookuptableimportv2_table_handle/
+key_value_init8291_lookuptableimportv2_keys1
-key_value_init8291_lookuptableimportv2_values	
identity??&key_value_init8291/LookupTableImportV2?
&key_value_init8291/LookupTableImportV2LookupTableImportV23key_value_init8291_lookuptableimportv2_table_handle+key_value_init8291_lookuptableimportv2_keys-key_value_init8291_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init8291/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init8291/LookupTableImportV2&key_value_init8291/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
'__inference_lstm_2_layer_call_fn_608260
inputs_0
unknown:	@?
	unknown_0:	?
	unknown_1:	d?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_606037o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?r
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_606439

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	%
embedding_1_606112:	?@ 
lstm_2_606390:	@?
lstm_2_606392:	? 
lstm_2_606394:	d?!
dense_2_606409:	d?
dense_2_606411:	?!
dense_3_606433:	?
dense_3_606435:
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
gtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ptext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountjtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????????????
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:??????????????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1_606112*
Tin
2	*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_606111X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
embedding_1/NotEqualNotEqual?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*0
_output_shapes
:???????????????????
lstm_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0embedding_1/NotEqual:z:0lstm_2_606390lstm_2_606392lstm_2_606394*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_606389?
dense_2/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_2_606409dense_2_606411*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_606408?
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_606419?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_606433dense_3_606435*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_606432w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_607294
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?@
	unknown_4:	@?
	unknown_5:	?
	unknown_6:	d?
	unknown_7:	d?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_605528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:?????????
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?F
?
__inference_adapt_step_607343
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
º
?
%sequential_1_lstm_2_while_body_605360D
@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counterJ
Fsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations)
%sequential_1_lstm_2_while_placeholder+
'sequential_1_lstm_2_while_placeholder_1+
'sequential_1_lstm_2_while_placeholder_2+
'sequential_1_lstm_2_while_placeholder_3+
'sequential_1_lstm_2_while_placeholder_4C
?sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1_0
{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0?
sequential_1_lstm_2_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0X
Esequential_1_lstm_2_while_lstm_cell_6_split_readvariableop_resource_0:	@?V
Gsequential_1_lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0:	?R
?sequential_1_lstm_2_while_lstm_cell_6_readvariableop_resource_0:	d?&
"sequential_1_lstm_2_while_identity(
$sequential_1_lstm_2_while_identity_1(
$sequential_1_lstm_2_while_identity_2(
$sequential_1_lstm_2_while_identity_3(
$sequential_1_lstm_2_while_identity_4(
$sequential_1_lstm_2_while_identity_5(
$sequential_1_lstm_2_while_identity_6A
=sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1}
ysequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor?
}sequential_1_lstm_2_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_1_tensorlistfromtensorV
Csequential_1_lstm_2_while_lstm_cell_6_split_readvariableop_resource:	@?T
Esequential_1_lstm_2_while_lstm_cell_6_split_1_readvariableop_resource:	?P
=sequential_1_lstm_2_while_lstm_cell_6_readvariableop_resource:	d???4sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp?6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_1?6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_2?6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_3?:sequential_1/lstm_2/while/lstm_cell_6/split/ReadVariableOp?<sequential_1/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp?
Ksequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
=sequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_2_while_placeholderTsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
Msequential_1/lstm_2/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
?sequential_1/lstm_2/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemsequential_1_lstm_2_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0%sequential_1_lstm_2_while_placeholderVsequential_1/lstm_2/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
5sequential_1/lstm_2/while/lstm_cell_6/ones_like/ShapeShapeDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:z
5sequential_1/lstm_2/while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
/sequential_1/lstm_2/while/lstm_cell_6/ones_likeFill>sequential_1/lstm_2/while/lstm_cell_6/ones_like/Shape:output:0>sequential_1/lstm_2/while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@?
7sequential_1/lstm_2/while/lstm_cell_6/ones_like_1/ShapeShape'sequential_1_lstm_2_while_placeholder_3*
T0*
_output_shapes
:|
7sequential_1/lstm_2/while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
1sequential_1/lstm_2/while/lstm_cell_6/ones_like_1Fill@sequential_1/lstm_2/while/lstm_cell_6/ones_like_1/Shape:output:0@sequential_1/lstm_2/while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
)sequential_1/lstm_2/while/lstm_cell_6/mulMulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_1/lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
+sequential_1/lstm_2/while/lstm_cell_6/mul_1MulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_1/lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
+sequential_1/lstm_2/while/lstm_cell_6/mul_2MulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_1/lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
+sequential_1/lstm_2/while/lstm_cell_6/mul_3MulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:08sequential_1/lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@w
5sequential_1/lstm_2/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
:sequential_1/lstm_2/while/lstm_cell_6/split/ReadVariableOpReadVariableOpEsequential_1_lstm_2_while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
+sequential_1/lstm_2/while/lstm_cell_6/splitSplit>sequential_1/lstm_2/while/lstm_cell_6/split/split_dim:output:0Bsequential_1/lstm_2/while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
,sequential_1/lstm_2/while/lstm_cell_6/MatMulMatMul-sequential_1/lstm_2/while/lstm_cell_6/mul:z:04sequential_1/lstm_2/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
.sequential_1/lstm_2/while/lstm_cell_6/MatMul_1MatMul/sequential_1/lstm_2/while/lstm_cell_6/mul_1:z:04sequential_1/lstm_2/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
.sequential_1/lstm_2/while/lstm_cell_6/MatMul_2MatMul/sequential_1/lstm_2/while/lstm_cell_6/mul_2:z:04sequential_1/lstm_2/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
.sequential_1/lstm_2/while/lstm_cell_6/MatMul_3MatMul/sequential_1/lstm_2/while/lstm_cell_6/mul_3:z:04sequential_1/lstm_2/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????dy
7sequential_1/lstm_2/while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential_1/lstm_2/while/lstm_cell_6/split_1/ReadVariableOpReadVariableOpGsequential_1_lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
-sequential_1/lstm_2/while/lstm_cell_6/split_1Split@sequential_1/lstm_2/while/lstm_cell_6/split_1/split_dim:output:0Dsequential_1/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
-sequential_1/lstm_2/while/lstm_cell_6/BiasAddBiasAdd6sequential_1/lstm_2/while/lstm_cell_6/MatMul:product:06sequential_1/lstm_2/while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
/sequential_1/lstm_2/while/lstm_cell_6/BiasAdd_1BiasAdd8sequential_1/lstm_2/while/lstm_cell_6/MatMul_1:product:06sequential_1/lstm_2/while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
/sequential_1/lstm_2/while/lstm_cell_6/BiasAdd_2BiasAdd8sequential_1/lstm_2/while/lstm_cell_6/MatMul_2:product:06sequential_1/lstm_2/while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
/sequential_1/lstm_2/while/lstm_cell_6/BiasAdd_3BiasAdd8sequential_1/lstm_2/while/lstm_cell_6/MatMul_3:product:06sequential_1/lstm_2/while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/mul_4Mul'sequential_1_lstm_2_while_placeholder_3:sequential_1/lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/mul_5Mul'sequential_1_lstm_2_while_placeholder_3:sequential_1/lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/mul_6Mul'sequential_1_lstm_2_while_placeholder_3:sequential_1/lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/mul_7Mul'sequential_1_lstm_2_while_placeholder_3:sequential_1/lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
4sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOpReadVariableOp?sequential_1_lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0?
9sequential_1/lstm_2/while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
;sequential_1/lstm_2/while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   ?
;sequential_1/lstm_2/while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
3sequential_1/lstm_2/while/lstm_cell_6/strided_sliceStridedSlice<sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp:value:0Bsequential_1/lstm_2/while/lstm_cell_6/strided_slice/stack:output:0Dsequential_1/lstm_2/while/lstm_cell_6/strided_slice/stack_1:output:0Dsequential_1/lstm_2/while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
.sequential_1/lstm_2/while/lstm_cell_6/MatMul_4MatMul/sequential_1/lstm_2/while/lstm_cell_6/mul_4:z:0<sequential_1/lstm_2/while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
)sequential_1/lstm_2/while/lstm_cell_6/addAddV26sequential_1/lstm_2/while/lstm_cell_6/BiasAdd:output:08sequential_1/lstm_2/while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????d?
-sequential_1/lstm_2/while/lstm_cell_6/SigmoidSigmoid-sequential_1/lstm_2/while/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_1ReadVariableOp?sequential_1_lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0?
;sequential_1/lstm_2/while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   ?
=sequential_1/lstm_2/while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   ?
=sequential_1/lstm_2/while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_2/while/lstm_cell_6/strided_slice_1StridedSlice>sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_1:value:0Dsequential_1/lstm_2/while/lstm_cell_6/strided_slice_1/stack:output:0Fsequential_1/lstm_2/while/lstm_cell_6/strided_slice_1/stack_1:output:0Fsequential_1/lstm_2/while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
.sequential_1/lstm_2/while/lstm_cell_6/MatMul_5MatMul/sequential_1/lstm_2/while/lstm_cell_6/mul_5:z:0>sequential_1/lstm_2/while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/add_1AddV28sequential_1/lstm_2/while/lstm_cell_6/BiasAdd_1:output:08sequential_1/lstm_2/while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????d?
/sequential_1/lstm_2/while/lstm_cell_6/Sigmoid_1Sigmoid/sequential_1/lstm_2/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/mul_8Mul3sequential_1/lstm_2/while/lstm_cell_6/Sigmoid_1:y:0'sequential_1_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:?????????d?
6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_2ReadVariableOp?sequential_1_lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0?
;sequential_1/lstm_2/while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
=sequential_1/lstm_2/while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  ?
=sequential_1/lstm_2/while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_2/while/lstm_cell_6/strided_slice_2StridedSlice>sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_2:value:0Dsequential_1/lstm_2/while/lstm_cell_6/strided_slice_2/stack:output:0Fsequential_1/lstm_2/while/lstm_cell_6/strided_slice_2/stack_1:output:0Fsequential_1/lstm_2/while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
.sequential_1/lstm_2/while/lstm_cell_6/MatMul_6MatMul/sequential_1/lstm_2/while/lstm_cell_6/mul_6:z:0>sequential_1/lstm_2/while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/add_2AddV28sequential_1/lstm_2/while/lstm_cell_6/BiasAdd_2:output:08sequential_1/lstm_2/while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????d?
*sequential_1/lstm_2/while/lstm_cell_6/TanhTanh/sequential_1/lstm_2/while/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/mul_9Mul1sequential_1/lstm_2/while/lstm_cell_6/Sigmoid:y:0.sequential_1/lstm_2/while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/add_3AddV2/sequential_1/lstm_2/while/lstm_cell_6/mul_8:z:0/sequential_1/lstm_2/while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_3ReadVariableOp?sequential_1_lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0?
;sequential_1/lstm_2/while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  ?
=sequential_1/lstm_2/while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
=sequential_1/lstm_2/while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
5sequential_1/lstm_2/while/lstm_cell_6/strided_slice_3StridedSlice>sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_3:value:0Dsequential_1/lstm_2/while/lstm_cell_6/strided_slice_3/stack:output:0Fsequential_1/lstm_2/while/lstm_cell_6/strided_slice_3/stack_1:output:0Fsequential_1/lstm_2/while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
.sequential_1/lstm_2/while/lstm_cell_6/MatMul_7MatMul/sequential_1/lstm_2/while/lstm_cell_6/mul_7:z:0>sequential_1/lstm_2/while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
+sequential_1/lstm_2/while/lstm_cell_6/add_4AddV28sequential_1/lstm_2/while/lstm_cell_6/BiasAdd_3:output:08sequential_1/lstm_2/while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????d?
/sequential_1/lstm_2/while/lstm_cell_6/Sigmoid_2Sigmoid/sequential_1/lstm_2/while/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????d?
,sequential_1/lstm_2/while/lstm_cell_6/Tanh_1Tanh/sequential_1/lstm_2/while/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
,sequential_1/lstm_2/while/lstm_cell_6/mul_10Mul3sequential_1/lstm_2/while/lstm_cell_6/Sigmoid_2:y:00sequential_1/lstm_2/while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dy
(sequential_1/lstm_2/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
sequential_1/lstm_2/while/TileTileFsequential_1/lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:01sequential_1/lstm_2/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
"sequential_1/lstm_2/while/SelectV2SelectV2'sequential_1/lstm_2/while/Tile:output:00sequential_1/lstm_2/while/lstm_cell_6/mul_10:z:0'sequential_1_lstm_2_while_placeholder_2*
T0*'
_output_shapes
:?????????d{
*sequential_1/lstm_2/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
 sequential_1/lstm_2/while/Tile_1TileFsequential_1/lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:03sequential_1/lstm_2/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????{
*sequential_1/lstm_2/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
 sequential_1/lstm_2/while/Tile_2TileFsequential_1/lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:03sequential_1/lstm_2/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:??????????
$sequential_1/lstm_2/while/SelectV2_1SelectV2)sequential_1/lstm_2/while/Tile_1:output:00sequential_1/lstm_2/while/lstm_cell_6/mul_10:z:0'sequential_1_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:?????????d?
$sequential_1/lstm_2/while/SelectV2_2SelectV2)sequential_1/lstm_2/while/Tile_2:output:0/sequential_1/lstm_2/while/lstm_cell_6/add_3:z:0'sequential_1_lstm_2_while_placeholder_4*
T0*'
_output_shapes
:?????????d?
Dsequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
>sequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_2_while_placeholder_1Msequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0+sequential_1/lstm_2/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???a
sequential_1/lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/lstm_2/while/addAddV2%sequential_1_lstm_2_while_placeholder(sequential_1/lstm_2/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_1/lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
sequential_1/lstm_2/while/add_1AddV2@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counter*sequential_1/lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: ?
"sequential_1/lstm_2/while/IdentityIdentity#sequential_1/lstm_2/while/add_1:z:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_2/while/Identity_1IdentityFsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_2/while/Identity_2Identity!sequential_1/lstm_2/while/add:z:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_2/while/Identity_3IdentityNsequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: ?
$sequential_1/lstm_2/while/Identity_4Identity+sequential_1/lstm_2/while/SelectV2:output:0^sequential_1/lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
$sequential_1/lstm_2/while/Identity_5Identity-sequential_1/lstm_2/while/SelectV2_1:output:0^sequential_1/lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
$sequential_1/lstm_2/while/Identity_6Identity-sequential_1/lstm_2/while/SelectV2_2:output:0^sequential_1/lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
sequential_1/lstm_2/while/NoOpNoOp5^sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp7^sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_17^sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_27^sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_3;^sequential_1/lstm_2/while/lstm_cell_6/split/ReadVariableOp=^sequential_1/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_1_lstm_2_while_identity+sequential_1/lstm_2/while/Identity:output:0"U
$sequential_1_lstm_2_while_identity_1-sequential_1/lstm_2/while/Identity_1:output:0"U
$sequential_1_lstm_2_while_identity_2-sequential_1/lstm_2/while/Identity_2:output:0"U
$sequential_1_lstm_2_while_identity_3-sequential_1/lstm_2/while/Identity_3:output:0"U
$sequential_1_lstm_2_while_identity_4-sequential_1/lstm_2/while/Identity_4:output:0"U
$sequential_1_lstm_2_while_identity_5-sequential_1/lstm_2/while/Identity_5:output:0"U
$sequential_1_lstm_2_while_identity_6-sequential_1/lstm_2/while/Identity_6:output:0"?
=sequential_1_lstm_2_while_lstm_cell_6_readvariableop_resource?sequential_1_lstm_2_while_lstm_cell_6_readvariableop_resource_0"?
Esequential_1_lstm_2_while_lstm_cell_6_split_1_readvariableop_resourceGsequential_1_lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0"?
Csequential_1_lstm_2_while_lstm_cell_6_split_readvariableop_resourceEsequential_1_lstm_2_while_lstm_cell_6_split_readvariableop_resource_0"?
=sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1?sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1_0"?
}sequential_1_lstm_2_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_1_tensorlistfromtensorsequential_1_lstm_2_while_tensorarrayv2read_1_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0"?
ysequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : 2l
4sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp4sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp2p
6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_16sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_12p
6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_26sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_22p
6sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_36sequential_1/lstm_2/while/lstm_cell_6/ReadVariableOp_32x
:sequential_1/lstm_2/while/lstm_cell_6/split/ReadVariableOp:sequential_1/lstm_2/while/lstm_cell_6/split/ReadVariableOp2|
<sequential_1/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp<sequential_1/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
??
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_608529
inputs_0<
)lstm_cell_6_split_readvariableop_resource:	@?:
+lstm_cell_6_split_1_readvariableop_resource:	?6
#lstm_cell_6_readvariableop_resource:	d?
identity??lstm_cell_6/ReadVariableOp?lstm_cell_6/ReadVariableOp_1?lstm_cell_6/ReadVariableOp_2?lstm_cell_6/ReadVariableOp_3? lstm_cell_6/split/ReadVariableOp?"lstm_cell_6/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskc
lstm_cell_6/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_likeFill$lstm_cell_6/ones_like/Shape:output:0$lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@[
lstm_cell_6/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_like_1Fill&lstm_cell_6/ones_like_1/Shape:output:0&lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/mulMulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_1Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_2Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_3Mulstrided_slice_2:output:0lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_6/split/ReadVariableOpReadVariableOp)lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0(lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split
lstm_cell_6/MatMulMatMullstm_cell_6/mul:z:0lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_1MatMullstm_cell_6/mul_1:z:0lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_2MatMullstm_cell_6/mul_2:z:0lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_3MatMullstm_cell_6/mul_3:z:0lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????d_
lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_6/split_1/ReadVariableOpReadVariableOp+lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_6/split_1Split&lstm_cell_6/split_1/split_dim:output:0*lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
lstm_cell_6/BiasAddBiasAddlstm_cell_6/MatMul:product:0lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_1BiasAddlstm_cell_6/MatMul_1:product:0lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_2BiasAddlstm_cell_6/MatMul_2:product:0lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_3BiasAddlstm_cell_6/MatMul_3:product:0lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_4Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_5Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_6Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d|
lstm_cell_6/mul_7Mulzeros:output:0 lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d
lstm_cell_6/ReadVariableOpReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0p
lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   r
!lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_sliceStridedSlice"lstm_cell_6/ReadVariableOp:value:0(lstm_cell_6/strided_slice/stack:output:0*lstm_cell_6/strided_slice/stack_1:output:0*lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_4MatMullstm_cell_6/mul_4:z:0"lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/addAddV2lstm_cell_6/BiasAdd:output:0lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????de
lstm_cell_6/SigmoidSigmoidlstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_1ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   t
#lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_1StridedSlice$lstm_cell_6/ReadVariableOp_1:value:0*lstm_cell_6/strided_slice_1/stack:output:0,lstm_cell_6/strided_slice_1/stack_1:output:0,lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_5MatMullstm_cell_6/mul_5:z:0$lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_1AddV2lstm_cell_6/BiasAdd_1:output:0lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????dw
lstm_cell_6/mul_8Mullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_2ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_2StridedSlice$lstm_cell_6/ReadVariableOp_2:value:0*lstm_cell_6/strided_slice_2/stack:output:0,lstm_cell_6/strided_slice_2/stack_1:output:0,lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_6MatMullstm_cell_6/mul_6:z:0$lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_2AddV2lstm_cell_6/BiasAdd_2:output:0lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????da
lstm_cell_6/TanhTanhlstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????dy
lstm_cell_6/mul_9Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????dz
lstm_cell_6/add_3AddV2lstm_cell_6/mul_8:z:0lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_3ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_3StridedSlice$lstm_cell_6/ReadVariableOp_3:value:0*lstm_cell_6/strided_slice_3/stack:output:0,lstm_cell_6/strided_slice_3/stack_1:output:0,lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_7MatMullstm_cell_6/mul_7:z:0$lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_4AddV2lstm_cell_6/BiasAdd_3:output:0lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????dc
lstm_cell_6/Tanh_1Tanhlstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d~
lstm_cell_6/mul_10Mullstm_cell_6/Sigmoid_2:y:0lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_6_split_readvariableop_resource+lstm_cell_6_split_1_readvariableop_resource#lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_608394*
condR
while_cond_608393*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^lstm_cell_6/ReadVariableOp^lstm_cell_6/ReadVariableOp_1^lstm_cell_6/ReadVariableOp_2^lstm_cell_6/ReadVariableOp_3!^lstm_cell_6/split/ReadVariableOp#^lstm_cell_6/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 28
lstm_cell_6/ReadVariableOplstm_cell_6/ReadVariableOp2<
lstm_cell_6/ReadVariableOp_1lstm_cell_6/ReadVariableOp_12<
lstm_cell_6/ReadVariableOp_2lstm_cell_6/ReadVariableOp_22<
lstm_cell_6/ReadVariableOp_3lstm_cell_6/ReadVariableOp_32D
 lstm_cell_6/split/ReadVariableOp lstm_cell_6/split/ReadVariableOp2H
"lstm_cell_6/split_1/ReadVariableOp"lstm_cell_6/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
'__inference_lstm_2_layer_call_fn_608249
inputs_0
unknown:	@?
	unknown_0:	?
	unknown_1:	d?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_605730o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
??
?
B__inference_lstm_2_layer_call_and_return_conditional_losses_608902
inputs_0<
)lstm_cell_6_split_readvariableop_resource:	@?:
+lstm_cell_6_split_1_readvariableop_resource:	?6
#lstm_cell_6_readvariableop_resource:	d?
identity??lstm_cell_6/ReadVariableOp?lstm_cell_6/ReadVariableOp_1?lstm_cell_6/ReadVariableOp_2?lstm_cell_6/ReadVariableOp_3? lstm_cell_6/split/ReadVariableOp?"lstm_cell_6/split_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????@D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????@*
shrink_axis_maskc
lstm_cell_6/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:`
lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_likeFill$lstm_cell_6/ones_like/Shape:output:0$lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@^
lstm_cell_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout/MulMullstm_cell_6/ones_like:output:0"lstm_cell_6/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@g
lstm_cell_6/dropout/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
0lstm_cell_6/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_6/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0g
"lstm_cell_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
 lstm_cell_6/dropout/GreaterEqualGreaterEqual9lstm_cell_6/dropout/random_uniform/RandomUniform:output:0+lstm_cell_6/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout/CastCast$lstm_cell_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout/Mul_1Mullstm_cell_6/dropout/Mul:z:0lstm_cell_6/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_1/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_1/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_1/GreaterEqualGreaterEqual;lstm_cell_6/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_1/CastCast&lstm_cell_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_1/Mul_1Mullstm_cell_6/dropout_1/Mul:z:0lstm_cell_6/dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_2/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_2/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_2/GreaterEqualGreaterEqual;lstm_cell_6/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_2/CastCast&lstm_cell_6/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_2/Mul_1Mullstm_cell_6/dropout_2/Mul:z:0lstm_cell_6/dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@`
lstm_cell_6/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_3/MulMullstm_cell_6/ones_like:output:0$lstm_cell_6/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@i
lstm_cell_6/dropout_3/ShapeShapelstm_cell_6/ones_like:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0i
$lstm_cell_6/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_3/GreaterEqualGreaterEqual;lstm_cell_6/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_3/CastCast&lstm_cell_6/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
lstm_cell_6/dropout_3/Mul_1Mullstm_cell_6/dropout_3/Mul:z:0lstm_cell_6/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@[
lstm_cell_6/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:b
lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/ones_like_1Fill&lstm_cell_6/ones_like_1/Shape:output:0&lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_4/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_4/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_4/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_4/GreaterEqualGreaterEqual;lstm_cell_6/dropout_4/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_4/CastCast&lstm_cell_6/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_4/Mul_1Mullstm_cell_6/dropout_4/Mul:z:0lstm_cell_6/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_5/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_5/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_5/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_5/GreaterEqualGreaterEqual;lstm_cell_6/dropout_5/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_5/CastCast&lstm_cell_6/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_5/Mul_1Mullstm_cell_6/dropout_5/Mul:z:0lstm_cell_6/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_6/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_6/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_6/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_6/GreaterEqualGreaterEqual;lstm_cell_6/dropout_6/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_6/CastCast&lstm_cell_6/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_6/Mul_1Mullstm_cell_6/dropout_6/Mul:z:0lstm_cell_6/dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????d`
lstm_cell_6/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
lstm_cell_6/dropout_7/MulMul lstm_cell_6/ones_like_1:output:0$lstm_cell_6/dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dk
lstm_cell_6/dropout_7/ShapeShape lstm_cell_6/ones_like_1:output:0*
T0*
_output_shapes
:?
2lstm_cell_6/dropout_7/random_uniform/RandomUniformRandomUniform$lstm_cell_6/dropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0i
$lstm_cell_6/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
"lstm_cell_6/dropout_7/GreaterEqualGreaterEqual;lstm_cell_6/dropout_7/random_uniform/RandomUniform:output:0-lstm_cell_6/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_7/CastCast&lstm_cell_6/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
lstm_cell_6/dropout_7/Mul_1Mullstm_cell_6/dropout_7/Mul:z:0lstm_cell_6/dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/mulMulstrided_slice_2:output:0lstm_cell_6/dropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_1Mulstrided_slice_2:output:0lstm_cell_6/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_2Mulstrided_slice_2:output:0lstm_cell_6/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@?
lstm_cell_6/mul_3Mulstrided_slice_2:output:0lstm_cell_6/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@]
lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
 lstm_cell_6/split/ReadVariableOpReadVariableOp)lstm_cell_6_split_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
lstm_cell_6/splitSplit$lstm_cell_6/split/split_dim:output:0(lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split
lstm_cell_6/MatMulMatMullstm_cell_6/mul:z:0lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_1MatMullstm_cell_6/mul_1:z:0lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_2MatMullstm_cell_6/mul_2:z:0lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/MatMul_3MatMullstm_cell_6/mul_3:z:0lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????d_
lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"lstm_cell_6/split_1/ReadVariableOpReadVariableOp+lstm_cell_6_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
lstm_cell_6/split_1Split&lstm_cell_6/split_1/split_dim:output:0*lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
lstm_cell_6/BiasAddBiasAddlstm_cell_6/MatMul:product:0lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_1BiasAddlstm_cell_6/MatMul_1:product:0lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_2BiasAddlstm_cell_6/MatMul_2:product:0lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/BiasAdd_3BiasAddlstm_cell_6/MatMul_3:product:0lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_4Mulzeros:output:0lstm_cell_6/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_5Mulzeros:output:0lstm_cell_6/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_6Mulzeros:output:0lstm_cell_6/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d{
lstm_cell_6/mul_7Mulzeros:output:0lstm_cell_6/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????d
lstm_cell_6/ReadVariableOpReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0p
lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   r
!lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_sliceStridedSlice"lstm_cell_6/ReadVariableOp:value:0(lstm_cell_6/strided_slice/stack:output:0*lstm_cell_6/strided_slice/stack_1:output:0*lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_4MatMullstm_cell_6/mul_4:z:0"lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/addAddV2lstm_cell_6/BiasAdd:output:0lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????de
lstm_cell_6/SigmoidSigmoidlstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_1ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   t
#lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_1StridedSlice$lstm_cell_6/ReadVariableOp_1:value:0*lstm_cell_6/strided_slice_1/stack:output:0,lstm_cell_6/strided_slice_1/stack_1:output:0,lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_5MatMullstm_cell_6/mul_5:z:0$lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_1AddV2lstm_cell_6/BiasAdd_1:output:0lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_1Sigmoidlstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????dw
lstm_cell_6/mul_8Mullstm_cell_6/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_2ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   t
#lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_2StridedSlice$lstm_cell_6/ReadVariableOp_2:value:0*lstm_cell_6/strided_slice_2/stack:output:0,lstm_cell_6/strided_slice_2/stack_1:output:0,lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_6MatMullstm_cell_6/mul_6:z:0$lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_2AddV2lstm_cell_6/BiasAdd_2:output:0lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????da
lstm_cell_6/TanhTanhlstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????dy
lstm_cell_6/mul_9Mullstm_cell_6/Sigmoid:y:0lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????dz
lstm_cell_6/add_3AddV2lstm_cell_6/mul_8:z:0lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/ReadVariableOp_3ReadVariableOp#lstm_cell_6_readvariableop_resource*
_output_shapes
:	d?*
dtype0r
!lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  t
#lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        t
#lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_cell_6/strided_slice_3StridedSlice$lstm_cell_6/ReadVariableOp_3:value:0*lstm_cell_6/strided_slice_3/stack:output:0,lstm_cell_6/strided_slice_3/stack_1:output:0,lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
lstm_cell_6/MatMul_7MatMullstm_cell_6/mul_7:z:0$lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_cell_6/add_4AddV2lstm_cell_6/BiasAdd_3:output:0lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????di
lstm_cell_6/Sigmoid_2Sigmoidlstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????dc
lstm_cell_6/Tanh_1Tanhlstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d~
lstm_cell_6/mul_10Mullstm_cell_6/Sigmoid_2:y:0lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lstm_cell_6_split_readvariableop_resource+lstm_cell_6_split_1_readvariableop_resource#lstm_cell_6_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????d:?????????d: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_608703*
condR
while_cond_608702*K
output_shapes:
8: : : : :?????????d:?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^lstm_cell_6/ReadVariableOp^lstm_cell_6/ReadVariableOp_1^lstm_cell_6/ReadVariableOp_2^lstm_cell_6/ReadVariableOp_3!^lstm_cell_6/split/ReadVariableOp#^lstm_cell_6/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????@: : : 28
lstm_cell_6/ReadVariableOplstm_cell_6/ReadVariableOp2<
lstm_cell_6/ReadVariableOp_1lstm_cell_6/ReadVariableOp_12<
lstm_cell_6/ReadVariableOp_2lstm_cell_6/ReadVariableOp_22<
lstm_cell_6/ReadVariableOp_3lstm_cell_6/ReadVariableOp_32D
 lstm_cell_6/split/ReadVariableOp lstm_cell_6/split/ReadVariableOp2H
"lstm_cell_6/split_1/ReadVariableOp"lstm_cell_6/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0
?
?
while_cond_605966
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_605966___redundant_placeholder04
0while_while_cond_605966___redundant_placeholder14
0while_while_cond_605966___redundant_placeholder24
0while_while_cond_605966___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?	
?
while_cond_609358
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice_14
0while_while_cond_609358___redundant_placeholder04
0while_while_cond_609358___redundant_placeholder14
0while_while_cond_609358___redundant_placeholder24
0while_while_cond_609358___redundant_placeholder34
0while_while_cond_609358___redundant_placeholder4
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :?????????d:?????????d:?????????d: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
?
while_cond_605659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_605659___redundant_placeholder04
0while_while_cond_605659___redundant_placeholder14
0while_while_cond_605659___redundant_placeholder24
0while_while_cond_605659___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
lstm_2_while_cond_607982*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3
lstm_2_while_placeholder_4,
(lstm_2_while_less_lstm_2_strided_slice_1B
>lstm_2_while_lstm_2_while_cond_607982___redundant_placeholder0B
>lstm_2_while_lstm_2_while_cond_607982___redundant_placeholder1B
>lstm_2_while_lstm_2_while_cond_607982___redundant_placeholder2B
>lstm_2_while_lstm_2_while_cond_607982___redundant_placeholder3B
>lstm_2_while_lstm_2_while_cond_607982___redundant_placeholder4
lstm_2_while_identity
~
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: Y
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :?????????d:?????????d:?????????d: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
?
%sequential_1_lstm_2_while_cond_605359D
@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counterJ
Fsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations)
%sequential_1_lstm_2_while_placeholder+
'sequential_1_lstm_2_while_placeholder_1+
'sequential_1_lstm_2_while_placeholder_2+
'sequential_1_lstm_2_while_placeholder_3+
'sequential_1_lstm_2_while_placeholder_4F
Bsequential_1_lstm_2_while_less_sequential_1_lstm_2_strided_slice_1\
Xsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_605359___redundant_placeholder0\
Xsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_605359___redundant_placeholder1\
Xsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_605359___redundant_placeholder2\
Xsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_605359___redundant_placeholder3\
Xsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_605359___redundant_placeholder4&
"sequential_1_lstm_2_while_identity
?
sequential_1/lstm_2/while/LessLess%sequential_1_lstm_2_while_placeholderBsequential_1_lstm_2_while_less_sequential_1_lstm_2_strided_slice_1*
T0*
_output_shapes
: s
"sequential_1/lstm_2/while/IdentityIdentity"sequential_1/lstm_2/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_1_lstm_2_while_identity+sequential_1/lstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :?????????d:?????????d:?????????d: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
::	

_output_shapes
:
?
F
*__inference_dropout_1_layer_call_fn_609601

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_606419a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
'__inference_lstm_2_layer_call_fn_608284

inputs
mask

unknown:	@?
	unknown_0:	?
	unknown_1:	d?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsmaskunknown	unknown_0	unknown_1*
Tin	
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lstm_2_layer_call_and_return_conditional_losses_606925o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????????????@:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:VR
0
_output_shapes
:??????????????????

_user_specified_namemask
ԙ
?
lstm_2_while_body_607576*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3
lstm_2_while_placeholder_4)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0i
elstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0K
8lstm_2_while_lstm_cell_6_split_readvariableop_resource_0:	@?I
:lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0:	?E
2lstm_2_while_lstm_cell_6_readvariableop_resource_0:	d?
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5
lstm_2_while_identity_6'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorg
clstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensorI
6lstm_2_while_lstm_cell_6_split_readvariableop_resource:	@?G
8lstm_2_while_lstm_cell_6_split_1_readvariableop_resource:	?C
0lstm_2_while_lstm_cell_6_readvariableop_resource:	d???'lstm_2/while/lstm_cell_6/ReadVariableOp?)lstm_2/while/lstm_cell_6/ReadVariableOp_1?)lstm_2/while/lstm_cell_6/ReadVariableOp_2?)lstm_2/while/lstm_cell_6/ReadVariableOp_3?-lstm_2/while/lstm_cell_6/split/ReadVariableOp?/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp?
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   ?
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????@*
element_dtype0?
@lstm_2/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
2lstm_2/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemelstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0lstm_2_while_placeholderIlstm_2/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0
?
(lstm_2/while/lstm_cell_6/ones_like/ShapeShape7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:m
(lstm_2/while/lstm_cell_6/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"lstm_2/while/lstm_cell_6/ones_likeFill1lstm_2/while/lstm_cell_6/ones_like/Shape:output:01lstm_2/while/lstm_cell_6/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@t
*lstm_2/while/lstm_cell_6/ones_like_1/ShapeShapelstm_2_while_placeholder_3*
T0*
_output_shapes
:o
*lstm_2/while/lstm_cell_6/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$lstm_2/while/lstm_cell_6/ones_like_1Fill3lstm_2/while/lstm_cell_6/ones_like_1/Shape:output:03lstm_2/while/lstm_cell_6/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mulMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_2/while/lstm_cell_6/mul_1Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_2/while/lstm_cell_6/mul_2Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@?
lstm_2/while/lstm_cell_6/mul_3Mul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0+lstm_2/while/lstm_cell_6/ones_like:output:0*
T0*'
_output_shapes
:?????????@j
(lstm_2/while/lstm_cell_6/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :?
-lstm_2/while/lstm_cell_6/split/ReadVariableOpReadVariableOp8lstm_2_while_lstm_cell_6_split_readvariableop_resource_0*
_output_shapes
:	@?*
dtype0?
lstm_2/while/lstm_cell_6/splitSplit1lstm_2/while/lstm_cell_6/split/split_dim:output:05lstm_2/while/lstm_cell_6/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split?
lstm_2/while/lstm_cell_6/MatMulMatMul lstm_2/while/lstm_cell_6/mul:z:0'lstm_2/while/lstm_cell_6/split:output:0*
T0*'
_output_shapes
:?????????d?
!lstm_2/while/lstm_cell_6/MatMul_1MatMul"lstm_2/while/lstm_cell_6/mul_1:z:0'lstm_2/while/lstm_cell_6/split:output:1*
T0*'
_output_shapes
:?????????d?
!lstm_2/while/lstm_cell_6/MatMul_2MatMul"lstm_2/while/lstm_cell_6/mul_2:z:0'lstm_2/while/lstm_cell_6/split:output:2*
T0*'
_output_shapes
:?????????d?
!lstm_2/while/lstm_cell_6/MatMul_3MatMul"lstm_2/while/lstm_cell_6/mul_3:z:0'lstm_2/while/lstm_cell_6/split:output:3*
T0*'
_output_shapes
:?????????dl
*lstm_2/while/lstm_cell_6/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ?
/lstm_2/while/lstm_cell_6/split_1/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype0?
 lstm_2/while/lstm_cell_6/split_1Split3lstm_2/while/lstm_cell_6/split_1/split_dim:output:07lstm_2/while/lstm_cell_6/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_split?
 lstm_2/while/lstm_cell_6/BiasAddBiasAdd)lstm_2/while/lstm_cell_6/MatMul:product:0)lstm_2/while/lstm_cell_6/split_1:output:0*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/BiasAdd_1BiasAdd+lstm_2/while/lstm_cell_6/MatMul_1:product:0)lstm_2/while/lstm_cell_6/split_1:output:1*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/BiasAdd_2BiasAdd+lstm_2/while/lstm_cell_6/MatMul_2:product:0)lstm_2/while/lstm_cell_6/split_1:output:2*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/BiasAdd_3BiasAdd+lstm_2/while/lstm_cell_6/MatMul_3:product:0)lstm_2/while/lstm_cell_6/split_1:output:3*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_4Mullstm_2_while_placeholder_3-lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_5Mullstm_2_while_placeholder_3-lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_6Mullstm_2_while_placeholder_3-lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_7Mullstm_2_while_placeholder_3-lstm_2/while/lstm_cell_6/ones_like_1:output:0*
T0*'
_output_shapes
:?????????d?
'lstm_2/while/lstm_cell_6/ReadVariableOpReadVariableOp2lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0}
,lstm_2/while/lstm_cell_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.lstm_2/while/lstm_cell_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   
.lstm_2/while/lstm_cell_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&lstm_2/while/lstm_cell_6/strided_sliceStridedSlice/lstm_2/while/lstm_cell_6/ReadVariableOp:value:05lstm_2/while/lstm_cell_6/strided_slice/stack:output:07lstm_2/while/lstm_cell_6/strided_slice/stack_1:output:07lstm_2/while/lstm_cell_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_6/MatMul_4MatMul"lstm_2/while/lstm_cell_6/mul_4:z:0/lstm_2/while/lstm_cell_6/strided_slice:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/addAddV2)lstm_2/while/lstm_cell_6/BiasAdd:output:0+lstm_2/while/lstm_cell_6/MatMul_4:product:0*
T0*'
_output_shapes
:?????????d
 lstm_2/while/lstm_cell_6/SigmoidSigmoid lstm_2/while/lstm_cell_6/add:z:0*
T0*'
_output_shapes
:?????????d?
)lstm_2/while/lstm_cell_6/ReadVariableOp_1ReadVariableOp2lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0
.lstm_2/while/lstm_cell_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   ?
0lstm_2/while/lstm_cell_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   ?
0lstm_2/while/lstm_cell_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_6/strided_slice_1StridedSlice1lstm_2/while/lstm_cell_6/ReadVariableOp_1:value:07lstm_2/while/lstm_cell_6/strided_slice_1/stack:output:09lstm_2/while/lstm_cell_6/strided_slice_1/stack_1:output:09lstm_2/while/lstm_cell_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_6/MatMul_5MatMul"lstm_2/while/lstm_cell_6/mul_5:z:01lstm_2/while/lstm_cell_6/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/add_1AddV2+lstm_2/while/lstm_cell_6/BiasAdd_1:output:0+lstm_2/while/lstm_cell_6/MatMul_5:product:0*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/Sigmoid_1Sigmoid"lstm_2/while/lstm_cell_6/add_1:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_8Mul&lstm_2/while/lstm_cell_6/Sigmoid_1:y:0lstm_2_while_placeholder_4*
T0*'
_output_shapes
:?????????d?
)lstm_2/while/lstm_cell_6/ReadVariableOp_2ReadVariableOp2lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0
.lstm_2/while/lstm_cell_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   ?
0lstm_2/while/lstm_cell_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  ?
0lstm_2/while/lstm_cell_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_6/strided_slice_2StridedSlice1lstm_2/while/lstm_cell_6/ReadVariableOp_2:value:07lstm_2/while/lstm_cell_6/strided_slice_2/stack:output:09lstm_2/while/lstm_cell_6/strided_slice_2/stack_1:output:09lstm_2/while/lstm_cell_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_6/MatMul_6MatMul"lstm_2/while/lstm_cell_6/mul_6:z:01lstm_2/while/lstm_cell_6/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/add_2AddV2+lstm_2/while/lstm_cell_6/BiasAdd_2:output:0+lstm_2/while/lstm_cell_6/MatMul_6:product:0*
T0*'
_output_shapes
:?????????d{
lstm_2/while/lstm_cell_6/TanhTanh"lstm_2/while/lstm_cell_6/add_2:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_9Mul$lstm_2/while/lstm_cell_6/Sigmoid:y:0!lstm_2/while/lstm_cell_6/Tanh:y:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/add_3AddV2"lstm_2/while/lstm_cell_6/mul_8:z:0"lstm_2/while/lstm_cell_6/mul_9:z:0*
T0*'
_output_shapes
:?????????d?
)lstm_2/while/lstm_cell_6/ReadVariableOp_3ReadVariableOp2lstm_2_while_lstm_cell_6_readvariableop_resource_0*
_output_shapes
:	d?*
dtype0
.lstm_2/while/lstm_cell_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  ?
0lstm_2/while/lstm_cell_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0lstm_2/while/lstm_cell_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
(lstm_2/while/lstm_cell_6/strided_slice_3StridedSlice1lstm_2/while/lstm_cell_6/ReadVariableOp_3:value:07lstm_2/while/lstm_cell_6/strided_slice_3/stack:output:09lstm_2/while/lstm_cell_6/strided_slice_3/stack_1:output:09lstm_2/while/lstm_cell_6/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_mask?
!lstm_2/while/lstm_cell_6/MatMul_7MatMul"lstm_2/while/lstm_cell_6/mul_7:z:01lstm_2/while/lstm_cell_6/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/add_4AddV2+lstm_2/while/lstm_cell_6/BiasAdd_3:output:0+lstm_2/while/lstm_cell_6/MatMul_7:product:0*
T0*'
_output_shapes
:?????????d?
"lstm_2/while/lstm_cell_6/Sigmoid_2Sigmoid"lstm_2/while/lstm_cell_6/add_4:z:0*
T0*'
_output_shapes
:?????????d}
lstm_2/while/lstm_cell_6/Tanh_1Tanh"lstm_2/while/lstm_cell_6/add_3:z:0*
T0*'
_output_shapes
:?????????d?
lstm_2/while/lstm_cell_6/mul_10Mul&lstm_2/while/lstm_cell_6/Sigmoid_2:y:0#lstm_2/while/lstm_cell_6/Tanh_1:y:0*
T0*'
_output_shapes
:?????????dl
lstm_2/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_2/while/TileTile9lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:0$lstm_2/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:??????????
lstm_2/while/SelectV2SelectV2lstm_2/while/Tile:output:0#lstm_2/while/lstm_cell_6/mul_10:z:0lstm_2_while_placeholder_2*
T0*'
_output_shapes
:?????????dn
lstm_2/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_2/while/Tile_1Tile9lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_2/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:?????????n
lstm_2/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      ?
lstm_2/while/Tile_2Tile9lstm_2/while/TensorArrayV2Read_1/TensorListGetItem:item:0&lstm_2/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:??????????
lstm_2/while/SelectV2_1SelectV2lstm_2/while/Tile_1:output:0#lstm_2/while/lstm_cell_6/mul_10:z:0lstm_2_while_placeholder_3*
T0*'
_output_shapes
:?????????d?
lstm_2/while/SelectV2_2SelectV2lstm_2/while/Tile_2:output:0"lstm_2/while/lstm_cell_6/add_3:z:0lstm_2_while_placeholder_4*
T0*'
_output_shapes
:?????????dy
7lstm_2/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ?
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1@lstm_2/while/TensorArrayV2Write/TensorListSetItem/index:output:0lstm_2/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:???T
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: n
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: ?
lstm_2/while/Identity_4Identitylstm_2/while/SelectV2:output:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
lstm_2/while/Identity_5Identity lstm_2/while/SelectV2_1:output:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
lstm_2/while/Identity_6Identity lstm_2/while/SelectV2_2:output:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:?????????d?
lstm_2/while/NoOpNoOp(^lstm_2/while/lstm_cell_6/ReadVariableOp*^lstm_2/while/lstm_cell_6/ReadVariableOp_1*^lstm_2/while/lstm_cell_6/ReadVariableOp_2*^lstm_2/while/lstm_cell_6/ReadVariableOp_3.^lstm_2/while/lstm_cell_6/split/ReadVariableOp0^lstm_2/while/lstm_cell_6/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0";
lstm_2_while_identity_6 lstm_2/while/Identity_6:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"f
0lstm_2_while_lstm_cell_6_readvariableop_resource2lstm_2_while_lstm_cell_6_readvariableop_resource_0"v
8lstm_2_while_lstm_cell_6_split_1_readvariableop_resource:lstm_2_while_lstm_cell_6_split_1_readvariableop_resource_0"r
6lstm_2_while_lstm_cell_6_split_readvariableop_resource8lstm_2_while_lstm_cell_6_split_readvariableop_resource_0"?
clstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensorelstm_2_while_tensorarrayv2read_1_tensorlistgetitem_lstm_2_tensorarrayunstack_1_tensorlistfromtensor_0"?
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :?????????d:?????????d:?????????d: : : : : : 2R
'lstm_2/while/lstm_cell_6/ReadVariableOp'lstm_2/while/lstm_cell_6/ReadVariableOp2V
)lstm_2/while/lstm_cell_6/ReadVariableOp_1)lstm_2/while/lstm_cell_6/ReadVariableOp_12V
)lstm_2/while/lstm_cell_6/ReadVariableOp_2)lstm_2/while/lstm_cell_6/ReadVariableOp_22V
)lstm_2/while/lstm_cell_6/ReadVariableOp_3)lstm_2/while/lstm_cell_6/ReadVariableOp_32^
-lstm_2/while/lstm_cell_6/split/ReadVariableOp-lstm_2/while/lstm_cell_6/split/ReadVariableOp2b
/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp/lstm_2/while/lstm_cell_6/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?~
?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_609905

inputs
states_0
states_10
split_readvariableop_resource:	@?.
split_1_readvariableop_resource:	?*
readvariableop_resource:	d?
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????@R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:?????????@O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:?????????@Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:?????????@T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:?????????@Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:?????????@T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????@Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????@I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????dT
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????dS
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????ds
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????do
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????dT
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????dS
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????ds
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????do
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????dT
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:?????????dS
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????ds
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????do
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:?????????dT
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:?????????dS
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:?
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????ds
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????do
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:?????????dW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@[
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:?????????@[
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:?????????@[
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :s
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	@?*
dtype0?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:@d:@d:@d:@d*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:?????????d_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:?????????d_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:?????????d_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:?????????dS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:d:d:d:d*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????dl
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????dl
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????dl
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????d]
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????d]
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????d]
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:?????????d]
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:?????????dg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    d   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????di
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????dh
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????dW
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????di
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????dh
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????dU
mul_9MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????dV
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:?????????di
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	d?*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ,  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:dd*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????dh
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????dQ
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:?????????dK
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????dZ
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????d[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:?????????dZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:?????????d:?????????d: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/1
?
?
,__inference_lstm_cell_6_layer_call_fn_609660

inputs
states_0
states_1
unknown:	@?
	unknown_0:	?
	unknown_1:	d?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????d:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_605645o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????dq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????@:?????????d:?????????d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/1
?	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_606496

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_608702
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_608702___redundant_placeholder04
0while_while_cond_608702___redundant_placeholder14
0while_while_cond_608702___redundant_placeholder24
0while_while_cond_608702___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :?????????d:?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?	
?
__inference_restore_fn_609966
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :N
IdentityIdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1"?	L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Y
text_vectorization_input=
*serving_default_text_vectorization_input:0?????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _random_generator
!cell
"
state_spec"
_tf_keras_rnn_layer
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_random_generator"
_tf_keras_layer
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias"
_tf_keras_layer
X
1
:2
;3
<4
)5
*6
87
98"
trackable_list_wrapper
X
0
:1
;2
<3
)4
*5
86
97"
trackable_list_wrapper
 "
trackable_list_wrapper
?
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Btrace_0
Ctrace_1
Dtrace_2
Etrace_32?
-__inference_sequential_1_layer_call_fn_606466
-__inference_sequential_1_layer_call_fn_607372
-__inference_sequential_1_layer_call_fn_607401
-__inference_sequential_1_layer_call_fn_607105?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zBtrace_0zCtrace_1zDtrace_2zEtrace_3
?
Ftrace_0
Gtrace_1
Htrace_2
Itrace_32?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607744
H__inference_sequential_1_layer_call_and_return_conditional_losses_608222
H__inference_sequential_1_layer_call_and_return_conditional_losses_607181
H__inference_sequential_1_layer_call_and_return_conditional_losses_607257?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zFtrace_0zGtrace_1zHtrace_2zItrace_3
?
J	capture_1
K	capture_2
L	capture_3B?
!__inference__wrapped_model_605528text_vectorization_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
?
Miter

Nbeta_1

Obeta_2
	Pdecay
Qlearning_ratem?)m?*m?8m?9m?:m?;m?<m?v?)v?*v?8v?9v?:v?;v?<v?"
	optimizer
,
Rserving_default"
signature_map
"
_generic_user_object
L
S	keras_api
Tlookup_table
Utoken_counts"
_tf_keras_layer
?
Vtrace_02?
__inference_adapt_step_607343?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zVtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
\trace_02?
,__inference_embedding_1_layer_call_fn_608229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z\trace_0
?
]trace_02?
G__inference_embedding_1_layer_call_and_return_conditional_losses_608238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z]trace_0
):'	?@2embedding_1/embeddings
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

^states
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
dtrace_0
etrace_1
ftrace_2
gtrace_32?
'__inference_lstm_2_layer_call_fn_608249
'__inference_lstm_2_layer_call_fn_608260
'__inference_lstm_2_layer_call_fn_608272
'__inference_lstm_2_layer_call_fn_608284?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zdtrace_0zetrace_1zftrace_2zgtrace_3
?
htrace_0
itrace_1
jtrace_2
ktrace_32?
B__inference_lstm_2_layer_call_and_return_conditional_losses_608529
B__inference_lstm_2_layer_call_and_return_conditional_losses_608902
B__inference_lstm_2_layer_call_and_return_conditional_losses_609175
B__inference_lstm_2_layer_call_and_return_conditional_losses_609576?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zhtrace_0zitrace_1zjtrace_2zktrace_3
"
_generic_user_object
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses
r_random_generator
s
state_size

:kernel
;recurrent_kernel
<bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
?
ytrace_02?
(__inference_dense_2_layer_call_fn_609585?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zytrace_0
?
ztrace_02?
C__inference_dense_2_layer_call_and_return_conditional_losses_609596?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zztrace_0
!:	d?2dense_2/kernel
:?2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_1_layer_call_fn_609601
*__inference_dropout_1_layer_call_fn_609606?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_1_layer_call_and_return_conditional_losses_609611
E__inference_dropout_1_layer_call_and_return_conditional_losses_609623?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_3_layer_call_fn_609632?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_3_layer_call_and_return_conditional_losses_609643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:	?2dense_3/kernel
:2dense_3/bias
,:*	@?2lstm_2/lstm_cell_6/kernel
6:4	d?2#lstm_2/lstm_cell_6/recurrent_kernel
&:$?2lstm_2/lstm_cell_6/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
J	capture_1
K	capture_2
L	capture_3B?
-__inference_sequential_1_layer_call_fn_606466text_vectorization_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
?
J	capture_1
K	capture_2
L	capture_3B?
-__inference_sequential_1_layer_call_fn_607372inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
?
J	capture_1
K	capture_2
L	capture_3B?
-__inference_sequential_1_layer_call_fn_607401inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
?
J	capture_1
K	capture_2
L	capture_3B?
-__inference_sequential_1_layer_call_fn_607105text_vectorization_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
?
J	capture_1
K	capture_2
L	capture_3B?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607744inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
?
J	capture_1
K	capture_2
L	capture_3B?
H__inference_sequential_1_layer_call_and_return_conditional_losses_608222inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
?
J	capture_1
K	capture_2
L	capture_3B?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607181text_vectorization_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
?
J	capture_1
K	capture_2
L	capture_3B?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607257text_vectorization_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
J	capture_1
K	capture_2
L	capture_3B?
$__inference_signature_wrapper_607294text_vectorization_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zJ	capture_1zK	capture_2zL	capture_3
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
?
?	capture_1B?
__inference_adapt_step_607343iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_embedding_1_layer_call_fn_608229inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_embedding_1_layer_call_and_return_conditional_losses_608238inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_lstm_2_layer_call_fn_608249inputs/0"?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_lstm_2_layer_call_fn_608260inputs/0"?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_lstm_2_layer_call_fn_608272inputsmask"?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_lstm_2_layer_call_fn_608284inputsmask"?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_lstm_2_layer_call_and_return_conditional_losses_608529inputs/0"?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_lstm_2_layer_call_and_return_conditional_losses_608902inputs/0"?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_lstm_2_layer_call_and_return_conditional_losses_609175inputsmask"?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_lstm_2_layer_call_and_return_conditional_losses_609576inputsmask"?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
,__inference_lstm_cell_6_layer_call_fn_609660
,__inference_lstm_cell_6_layer_call_fn_609677?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_609759
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_609905?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_2_layer_call_fn_609585inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_2_layer_call_and_return_conditional_losses_609596inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dropout_1_layer_call_fn_609601inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
*__inference_dropout_1_layer_call_fn_609606inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dropout_1_layer_call_and_return_conditional_losses_609611inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dropout_1_layer_call_and_return_conditional_losses_609623inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_3_layer_call_fn_609632inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_3_layer_call_and_return_conditional_losses_609643inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
?
?trace_02?
__inference__creator_609910?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_609918?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_609923?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__creator_609928?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_609933?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_609938?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
!J	
Const_2jtf.TrackableConstant
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_lstm_cell_6_layer_call_fn_609660inputsstates/0states/1"?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
,__inference_lstm_cell_6_layer_call_fn_609677inputsstates/0states/1"?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_609759inputsstates/0states/1"?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_609905inputsstates/0states/1"?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?B?
__inference__creator_609910"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?
?	capture_1
?	capture_2B?
__inference__initializer_609918"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?	capture_1z?	capture_2
?B?
__inference__destroyer_609923"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_609928"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_609933"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_609938"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
.:,	?@2Adam/embedding_1/embeddings/m
&:$	d?2Adam/dense_2/kernel/m
 :?2Adam/dense_2/bias/m
&:$	?2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
1:/	@?2 Adam/lstm_2/lstm_cell_6/kernel/m
;:9	d?2*Adam/lstm_2/lstm_cell_6/recurrent_kernel/m
+:)?2Adam/lstm_2/lstm_cell_6/bias/m
.:,	?@2Adam/embedding_1/embeddings/v
&:$	d?2Adam/dense_2/kernel/v
 :?2Adam/dense_2/bias/v
&:$	?2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
1:/	@?2 Adam/lstm_2/lstm_cell_6/kernel/v
;:9	d?2*Adam/lstm_2/lstm_cell_6/recurrent_kernel/v
+:)?2Adam/lstm_2/lstm_cell_6/bias/v
?B?
__inference_save_fn_609957checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_609966restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	7
__inference__creator_609910?

? 
? "? 7
__inference__creator_609928?

? 
? "? 9
__inference__destroyer_609923?

? 
? "? 9
__inference__destroyer_609938?

? 
? "? B
__inference__initializer_609918T???

? 
? "? ;
__inference__initializer_609933?

? 
? "? ?
!__inference__wrapped_model_605528?TJKL:<;)*89=?:
3?0
.?+
text_vectorization_input?????????
? "1?.
,
dense_3!?
dense_3?????????k
__inference_adapt_step_607343JU???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
C__inference_dense_2_layer_call_and_return_conditional_losses_609596])*/?,
%?"
 ?
inputs?????????d
? "&?#
?
0??????????
? |
(__inference_dense_2_layer_call_fn_609585P)*/?,
%?"
 ?
inputs?????????d
? "????????????
C__inference_dense_3_layer_call_and_return_conditional_losses_609643]890?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_3_layer_call_fn_609632P890?-
&?#
!?
inputs??????????
? "???????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_609611^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_609623^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_1_layer_call_fn_609601Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_1_layer_call_fn_609606Q4?1
*?'
!?
inputs??????????
p
? "????????????
G__inference_embedding_1_layer_call_and_return_conditional_losses_608238q8?5
.?+
)?&
inputs??????????????????	
? "2?/
(?%
0??????????????????@
? ?
,__inference_embedding_1_layer_call_fn_608229d8?5
.?+
)?&
inputs??????????????????	
? "%?"??????????????????@?
B__inference_lstm_2_layer_call_and_return_conditional_losses_608529}:<;O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "%?"
?
0?????????d
? ?
B__inference_lstm_2_layer_call_and_return_conditional_losses_608902}:<;O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "%?"
?
0?????????d
? ?
B__inference_lstm_2_layer_call_and_return_conditional_losses_609175?:<;m?j
c?`
-?*
inputs??????????????????@
'?$
mask??????????????????

p 

 
? "%?"
?
0?????????d
? ?
B__inference_lstm_2_layer_call_and_return_conditional_losses_609576?:<;m?j
c?`
-?*
inputs??????????????????@
'?$
mask??????????????????

p

 
? "%?"
?
0?????????d
? ?
'__inference_lstm_2_layer_call_fn_608249p:<;O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p 

 
? "??????????d?
'__inference_lstm_2_layer_call_fn_608260p:<;O?L
E?B
4?1
/?,
inputs/0??????????????????@

 
p

 
? "??????????d?
'__inference_lstm_2_layer_call_fn_608272?:<;m?j
c?`
-?*
inputs??????????????????@
'?$
mask??????????????????

p 

 
? "??????????d?
'__inference_lstm_2_layer_call_fn_608284?:<;m?j
c?`
-?*
inputs??????????????????@
'?$
mask??????????????????

p

 
? "??????????d?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_609759?:<;??}
v?s
 ?
inputs?????????@
K?H
"?
states/0?????????d
"?
states/1?????????d
p 
? "s?p
i?f
?
0/0?????????d
E?B
?
0/1/0?????????d
?
0/1/1?????????d
? ?
G__inference_lstm_cell_6_layer_call_and_return_conditional_losses_609905?:<;??}
v?s
 ?
inputs?????????@
K?H
"?
states/0?????????d
"?
states/1?????????d
p
? "s?p
i?f
?
0/0?????????d
E?B
?
0/1/0?????????d
?
0/1/1?????????d
? ?
,__inference_lstm_cell_6_layer_call_fn_609660?:<;??}
v?s
 ?
inputs?????????@
K?H
"?
states/0?????????d
"?
states/1?????????d
p 
? "c?`
?
0?????????d
A?>
?
1/0?????????d
?
1/1?????????d?
,__inference_lstm_cell_6_layer_call_fn_609677?:<;??}
v?s
 ?
inputs?????????@
K?H
"?
states/0?????????d
"?
states/1?????????d
p
? "c?`
?
0?????????d
A?>
?
1/0?????????d
?
1/1?????????dz
__inference_restore_fn_609966YUK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_609957?U&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607181|TJKL:<;)*89E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607257|TJKL:<;)*89E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_607744jTJKL:<;)*893?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_608222jTJKL:<;)*893?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_1_layer_call_fn_606466oTJKL:<;)*89E?B
;?8
.?+
text_vectorization_input?????????
p 

 
? "???????????
-__inference_sequential_1_layer_call_fn_607105oTJKL:<;)*89E?B
;?8
.?+
text_vectorization_input?????????
p

 
? "???????????
-__inference_sequential_1_layer_call_fn_607372]TJKL:<;)*893?0
)?&
?
inputs?????????
p 

 
? "???????????
-__inference_sequential_1_layer_call_fn_607401]TJKL:<;)*893?0
)?&
?
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_607294?TJKL:<;)*89Y?V
? 
O?L
J
text_vectorization_input.?+
text_vectorization_input?????????"1?.
,
dense_3!?
dense_3?????????