

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
-subB&
$
	full_text

%7 = sub i32 %1, %2
-andB&
$
	full_text

%8 = and i32 %7, 31
"i32B

	full_text


i32 %7
,shlB%
#
	full_text

%9 = shl i32 1, %8
"i32B

	full_text


i32 %8
-shlB&
$
	full_text

%10 = shl i32 %9, 1
"i32B

	full_text


i32 %9
.addB'
%
	full_text

%11 = add i32 %9, -1
"i32B

	full_text


i32 %9
/andB(
&
	full_text

%12 = and i32 %11, %6
#i32B

	full_text
	
i32 %11
"i32B

	full_text


i32 %6
0lshrB(
&
	full_text

%13 = lshr i32 %6, %8
"i32B

	full_text


i32 %6
"i32B

	full_text


i32 %8
0mulB)
'
	full_text

%14 = mul i32 %10, %13
#i32B

	full_text
	
i32 %10
#i32B

	full_text
	
i32 %13
0addB)
'
	full_text

%15 = add i32 %14, %12
#i32B

	full_text
	
i32 %14
#i32B

	full_text
	
i32 %12
/addB(
&
	full_text

%16 = add i32 %15, %9
#i32B

	full_text
	
i32 %15
"i32B

	full_text


i32 %9
4zextB,
*
	full_text

%17 = zext i32 %15 to i64
#i32B

	full_text
	
i32 %15
VgetelementptrBE
C
	full_text6
4
2%18 = getelementptr inbounds i32, i32* %0, i64 %17
#i64B

	full_text
	
i64 %17
FloadB>
<
	full_text/
-
+%19 = load i32, i32* %18, align 4, !tbaa !8
%i32*B

	full_text


i32* %18
4zextB,
*
	full_text

%20 = zext i32 %16 to i64
#i32B

	full_text
	
i32 %16
VgetelementptrBE
C
	full_text6
4
2%21 = getelementptr inbounds i32, i32* %0, i64 %20
#i64B

	full_text
	
i64 %20
FloadB>
<
	full_text/
-
+%22 = load i32, i32* %21, align 4, !tbaa !8
%i32*B

	full_text


i32* %21
.andB'
%
	full_text

%23 = and i32 %1, 31
.shlB'
%
	full_text

%24 = shl i32 1, %23
#i32B

	full_text
	
i32 %23
/andB(
&
	full_text

%25 = and i32 %24, %6
#i32B

	full_text
	
i32 %24
"i32B

	full_text


i32 %6
3icmpB+
)
	full_text

%26 = icmp eq i32 %25, 0
#i32B

	full_text
	
i32 %25
-subB&
$
	full_text

%27 = sub i32 1, %3
AselectB7
5
	full_text(
&
$%28 = select i1 %26, i32 %3, i32 %27
!i1B

	full_text


i1 %26
#i32B

	full_text
	
i32 %27
6icmpB.
,
	full_text

%29 = icmp ugt i32 %19, %22
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %22
BselectB8
6
	full_text)
'
%%30 = select i1 %29, i32 %19, i32 %22
!i1B

	full_text


i1 %29
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %22
BselectB8
6
	full_text)
'
%%31 = select i1 %29, i32 %22, i32 %19
!i1B

	full_text


i1 %29
#i32B

	full_text
	
i32 %22
#i32B

	full_text
	
i32 %19
3icmpB+
)
	full_text

%32 = icmp eq i32 %28, 0
#i32B

	full_text
	
i32 %28
BselectB8
6
	full_text)
'
%%33 = select i1 %32, i32 %30, i32 %31
!i1B

	full_text


i1 %32
#i32B

	full_text
	
i32 %30
#i32B

	full_text
	
i32 %31
BselectB8
6
	full_text)
'
%%34 = select i1 %32, i32 %31, i32 %30
!i1B

	full_text


i1 %32
#i32B

	full_text
	
i32 %31
#i32B

	full_text
	
i32 %30
FstoreB=
;
	full_text.
,
*store i32 %33, i32* %18, align 4, !tbaa !8
#i32B

	full_text
	
i32 %33
%i32*B

	full_text


i32* %18
FstoreB=
;
	full_text.
,
*store i32 %34, i32* %21, align 4, !tbaa !8
#i32B

	full_text
	
i32 %34
%i32*B

	full_text


i32* %21
"retB

	full_text


ret void
&i32*8B

	full_text
	
i32* %0
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %2
-; undefined function B

	full_text

 
$i328B

	full_text


i32 31
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 1        	
 		                       !    "# "" $% $$ &' && (( )* )) +, +- ++ ./ .. 00 12 13 11 45 46 44 78 79 7: 77 ;< ;= ;> ;; ?@ ?? AB AC AD AA EF EG EH EE IJ IK II LM LN LL OP P $Q 0Q 1R R (S     
     	         ! #" %$ '( *) , -+ /. 20 3  5& 64 8  9& :4 <& =  >1 @? B7 C; D? F; G7 HA J KE M$ N O TT TT U U (V V .V ?W X X 	X )X 0"
bitonicSort"
_Z13get_global_idj*?
BitonicSort_Kernels.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes
??

wgsize_log1p
A?<A
 
transfer_bytes_log1p
A?<A

wgsize
?

devmap_label
 